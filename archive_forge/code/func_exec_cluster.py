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
def exec_cluster(config_file: str, *, cmd: Optional[str]=None, run_env: str='auto', screen: bool=False, tmux: bool=False, stop: bool=False, start: bool=False, override_cluster_name: Optional[str]=None, no_config_cache: bool=False, port_forward: Optional[Port_forward]=None, with_output: bool=False, _allow_uninitialized_state: bool=False, extra_screen_args: Optional[str]=None) -> str:
    """Runs a command on the specified cluster.

    Arguments:
        config_file: path to the cluster yaml
        cmd: command to run
        run_env: whether to run the command on the host or in a container.
            Select between "auto", "host" and "docker"
        screen: whether to run in a screen
        extra_screen_args: optional custom additional args to screen command
        tmux: whether to run in a tmux session
        stop: whether to stop the cluster after command run
        start: whether to start the cluster if it isn't up
        override_cluster_name: set the name of the cluster
        port_forward ( (int, int) or list[(int, int)] ): port(s) to forward
        _allow_uninitialized_state: whether to execute on an uninitialized head
            node.
    """
    assert not (screen and tmux), 'Can specify only one of `screen` or `tmux`.'
    assert run_env in RUN_ENV_TYPES, '--run_env must be in {}'.format(RUN_ENV_TYPES)
    cmd_output_util.set_allow_interactive(True)
    config = yaml.safe_load(open(config_file).read())
    if override_cluster_name is not None:
        config['cluster_name'] = override_cluster_name
    config = _bootstrap_config(config, no_config_cache=no_config_cache)
    head_node = _get_running_head_node(config, config_file, override_cluster_name, create_if_needed=start, _allow_uninitialized_state=_allow_uninitialized_state)
    provider = _get_node_provider(config['provider'], config['cluster_name'])
    updater = NodeUpdaterThread(node_id=head_node, provider_config=config['provider'], provider=provider, auth_config=config['auth'], cluster_name=config['cluster_name'], file_mounts=config['file_mounts'], initialization_commands=[], setup_commands=[], ray_start_commands=[], runtime_hash='', file_mounts_contents_hash='', is_head_node=True, rsync_options={'rsync_exclude': config.get('rsync_exclude'), 'rsync_filter': config.get('rsync_filter')}, docker_config=config.get('docker'))
    shutdown_after_run = False
    if cmd and stop:
        cmd = '; '.join([cmd, 'ray stop', 'ray teardown ~/ray_bootstrap_config.yaml --yes --workers-only'])
        shutdown_after_run = True
    result = _exec(updater, cmd, screen, tmux, port_forward=port_forward, with_output=with_output, run_env=run_env, shutdown_after_run=shutdown_after_run, extra_screen_args=extra_screen_args)
    if tmux or screen:
        attach_command_parts = ['ray attach', config_file]
        if override_cluster_name is not None:
            attach_command_parts.append('--cluster-name={}'.format(override_cluster_name))
        if tmux:
            attach_command_parts.append('--tmux')
        elif screen:
            attach_command_parts.append('--screen')
        attach_command = ' '.join(attach_command_parts)
        cli_logger.print('Run `{}` to check command status.', cf.bold(attach_command))
    return result