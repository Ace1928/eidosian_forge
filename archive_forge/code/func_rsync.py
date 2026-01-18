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
def rsync(config_file: str, source: Optional[str], target: Optional[str], override_cluster_name: Optional[str], down: bool, ip_address: Optional[str]=None, use_internal_ip: bool=False, no_config_cache: bool=False, all_nodes: bool=False, should_bootstrap: bool=True, _runner: ModuleType=subprocess) -> None:
    """Rsyncs files.

    Arguments:
        config_file: path to the cluster yaml
        source: source dir
        target: target dir
        override_cluster_name: set the name of the cluster
        down: whether we're syncing remote -> local
        ip_address: Address of node. Raise Exception
            if both ip_address and 'all_nodes' are provided.
        use_internal_ip: Whether the provided ip_address is
            public or private.
        all_nodes: whether to sync worker nodes in addition to the head node
        should_bootstrap: whether to bootstrap cluster config before syncing
    """
    if bool(source) != bool(target):
        cli_logger.abort('Expected either both a source and a target, or neither.')
    assert bool(source) == bool(target), 'Must either provide both or neither source and target.'
    if ip_address and all_nodes:
        cli_logger.abort("Cannot provide both ip_address and 'all_nodes'.")
    config = yaml.safe_load(open(config_file).read())
    if override_cluster_name is not None:
        config['cluster_name'] = override_cluster_name
    if should_bootstrap:
        config = _bootstrap_config(config, no_config_cache=no_config_cache)
    is_file_mount = False
    if source and target:
        for remote_mount in config.get('file_mounts', {}).keys():
            if (source if down else target).startswith(remote_mount):
                is_file_mount = True
                break
    provider = _get_node_provider(config['provider'], config['cluster_name'])

    def rsync_to_node(node_id, is_head_node):
        updater = NodeUpdaterThread(node_id=node_id, provider_config=config['provider'], provider=provider, auth_config=config['auth'], cluster_name=config['cluster_name'], file_mounts=config['file_mounts'], initialization_commands=[], setup_commands=[], ray_start_commands=[], runtime_hash='', use_internal_ip=use_internal_ip, process_runner=_runner, file_mounts_contents_hash='', is_head_node=is_head_node, rsync_options={'rsync_exclude': config.get('rsync_exclude'), 'rsync_filter': config.get('rsync_filter')}, docker_config=config.get('docker'))
        if down:
            rsync = updater.rsync_down
        else:
            rsync = updater.rsync_up
        if source and target:
            if cli_logger.verbosity > 0:
                cmd_output_util.set_output_redirected(False)
                set_rsync_silent(False)
            rsync(source, target, is_file_mount)
        else:
            updater.sync_file_mounts(rsync)
    nodes = []
    head_node = _get_running_head_node(config, config_file, override_cluster_name, create_if_needed=False)
    if ip_address:
        nodes = [provider.get_node_id(ip_address, use_internal_ip=use_internal_ip)]
    else:
        nodes = [head_node]
        if all_nodes:
            nodes.extend(_get_worker_nodes(config, override_cluster_name))
    for node_id in nodes:
        rsync_to_node(node_id, is_head_node=node_id == head_node)