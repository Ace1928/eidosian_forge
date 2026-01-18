import copy
import datetime
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union
import click
import yaml
import ray
from ray._private.usage import usage_lib
from ray.autoscaler._private import subprocess_output_util as cmd_output_util
from ray.autoscaler._private.autoscaler import AutoscalerSummary
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.cluster_dump import (
from ray.autoscaler._private.command_runner import (
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.log_timer import LogTimer
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.providers import (
from ray.autoscaler._private.updater import NodeUpdaterThread
from ray.autoscaler._private.util import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
from ray.experimental.internal_kv import _internal_kv_put, internal_kv_get_gcs_client
from ray.util.debug import log_once
def get_or_create_head_node(config: Dict[str, Any], printable_config_file: str, no_restart: bool, restart_only: bool, yes: bool, override_cluster_name: Optional[str], no_monitor_on_head: bool=False, _provider: Optional[NodeProvider]=None, _runner: ModuleType=subprocess) -> None:
    """Create the cluster head node, which in turn creates the workers."""
    global_event_system.execute_callback(CreateClusterEvent.cluster_booting_started)
    provider = _provider or _get_node_provider(config['provider'], config['cluster_name'])
    config = copy.deepcopy(config)
    head_node_tags = {TAG_RAY_NODE_KIND: NODE_KIND_HEAD}
    nodes = provider.non_terminated_nodes(head_node_tags)
    if len(nodes) > 0:
        head_node = nodes[0]
    else:
        head_node = None
    if not head_node:
        cli_logger.confirm(yes, 'No head node found. Launching a new cluster.', _abort=True)
        cli_logger.newline()
        usage_lib.show_usage_stats_prompt(cli=True)
    if head_node:
        if restart_only:
            cli_logger.confirm(yes, 'Updating cluster configuration and restarting the cluster Ray runtime. Setup commands will not be run due to `{}`.\n', cf.bold('--restart-only'), _abort=True)
            cli_logger.newline()
            usage_lib.show_usage_stats_prompt(cli=True)
        elif no_restart:
            cli_logger.print('Cluster Ray runtime will not be restarted due to `{}`.', cf.bold('--no-restart'))
            cli_logger.confirm(yes, 'Updating cluster configuration and running setup commands.', _abort=True)
        else:
            cli_logger.print('Updating cluster configuration and running full setup.')
            cli_logger.confirm(yes, cf.bold('Cluster Ray runtime will be restarted.'), _abort=True)
            cli_logger.newline()
            usage_lib.show_usage_stats_prompt(cli=True)
    cli_logger.newline()
    head_node_config = copy.deepcopy(config.get('head_node', {}))
    head_node_resources = None
    head_node_labels = None
    head_node_type = config.get('head_node_type')
    if head_node_type:
        head_node_tags[TAG_RAY_USER_NODE_TYPE] = head_node_type
        head_config = config['available_node_types'][head_node_type]
        head_node_config.update(head_config['node_config'])
        head_node_resources = head_config.get('resources')
        head_node_labels = head_config.get('labels')
    launch_hash = hash_launch_conf(head_node_config, config['auth'])
    creating_new_head = _should_create_new_head(head_node, launch_hash, head_node_type, provider)
    if creating_new_head:
        with cli_logger.group('Acquiring an up-to-date head node'):
            global_event_system.execute_callback(CreateClusterEvent.acquiring_new_head_node)
            if head_node is not None:
                cli_logger.confirm(yes, 'Relaunching the head node.', _abort=True)
                provider.terminate_node(head_node)
                cli_logger.print('Terminated head node {}', head_node)
            head_node_tags[TAG_RAY_LAUNCH_CONFIG] = launch_hash
            head_node_tags[TAG_RAY_NODE_NAME] = 'ray-{}-head'.format(config['cluster_name'])
            head_node_tags[TAG_RAY_NODE_STATUS] = STATUS_UNINITIALIZED
            provider.create_node(head_node_config, head_node_tags, 1)
            cli_logger.print('Launched a new head node')
            start = time.time()
            head_node = None
            with cli_logger.group('Fetching the new head node'):
                while True:
                    if time.time() - start > 50:
                        cli_logger.abort('Head node fetch timed out. Failed to create head node.')
                    nodes = provider.non_terminated_nodes(head_node_tags)
                    if len(nodes) == 1:
                        head_node = nodes[0]
                        break
                    time.sleep(POLL_INTERVAL)
            cli_logger.newline()
    global_event_system.execute_callback(CreateClusterEvent.head_node_acquired)
    with cli_logger.group('Setting up head node', _numbered=('<>', 1, 1), _tags=dict()):
        runtime_hash, file_mounts_contents_hash = hash_runtime_conf(config['file_mounts'], None, config)
        if not no_monitor_on_head:
            config, remote_config_file = _set_up_config_for_head_node(config, provider, no_restart)
            cli_logger.print('Prepared bootstrap config')
        if restart_only:
            if config.get('docker', {}).get('container_name'):
                setup_commands = config['head_setup_commands']
            else:
                setup_commands = []
            ray_start_commands = config['head_start_ray_commands']
        elif no_restart and (not creating_new_head):
            setup_commands = config['head_setup_commands']
            ray_start_commands = []
        else:
            setup_commands = config['head_setup_commands']
            ray_start_commands = config['head_start_ray_commands']
        if not no_restart:
            warn_about_bad_start_command(ray_start_commands, no_monitor_on_head)
        updater = NodeUpdaterThread(node_id=head_node, provider_config=config['provider'], provider=provider, auth_config=config['auth'], cluster_name=config['cluster_name'], file_mounts=config['file_mounts'], initialization_commands=config['initialization_commands'], setup_commands=setup_commands, ray_start_commands=ray_start_commands, process_runner=_runner, runtime_hash=runtime_hash, file_mounts_contents_hash=file_mounts_contents_hash, is_head_node=True, node_resources=head_node_resources, node_labels=head_node_labels, rsync_options={'rsync_exclude': config.get('rsync_exclude'), 'rsync_filter': config.get('rsync_filter')}, docker_config=config.get('docker'), restart_only=restart_only)
        updater.start()
        updater.join()
        provider.non_terminated_nodes(head_node_tags)
        if updater.exitcode != 0:
            cli_logger.abort('Failed to setup head node.')
            sys.exit(1)
    global_event_system.execute_callback(CreateClusterEvent.cluster_booting_completed, {'head_node_id': head_node})
    monitor_str = 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'
    if override_cluster_name:
        modifiers = ' --cluster-name={}'.format(quote(override_cluster_name))
    else:
        modifiers = ''
    cli_logger.newline()
    with cli_logger.group('Useful commands:'):
        printable_config_file = os.path.abspath(printable_config_file)
        cli_logger.print('To terminate the cluster:')
        cli_logger.print(cf.bold(f'  ray down {printable_config_file}{modifiers}'))
        cli_logger.newline()
        cli_logger.print('To retrieve the IP address of the cluster head:')
        cli_logger.print(cf.bold(f'  ray get-head-ip {printable_config_file}{modifiers}'))
        cli_logger.newline()
        cli_logger.print("To port-forward the cluster's Ray Dashboard to the local machine:")
        cli_logger.print(cf.bold(f'  ray dashboard {printable_config_file}{modifiers}'))
        cli_logger.newline()
        cli_logger.print('To submit a job to the cluster, port-forward the Ray Dashboard in another terminal and run:')
        cli_logger.print(cf.bold('  ray job submit --address http://localhost:<dashboard-port> --working-dir . -- python my_script.py'))
        cli_logger.newline()
        cli_logger.print('To connect to a terminal on the cluster head for debugging:')
        cli_logger.print(cf.bold(f'  ray attach {printable_config_file}{modifiers}'))
        cli_logger.newline()
        cli_logger.print('To monitor autoscaling:')
        cli_logger.print(cf.bold(f'  ray exec {printable_config_file}{modifiers} {quote(monitor_str)}'))
        cli_logger.newline()