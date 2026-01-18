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
def create_or_update_cluster(config_file: str, override_min_workers: Optional[int], override_max_workers: Optional[int], no_restart: bool, restart_only: bool, yes: bool, override_cluster_name: Optional[str]=None, no_config_cache: bool=False, redirect_command_output: Optional[bool]=False, use_login_shells: bool=True, no_monitor_on_head: bool=False) -> Dict[str, Any]:
    """Creates or updates an autoscaling Ray cluster from a config json."""
    set_using_login_shells(use_login_shells)
    if not use_login_shells:
        cmd_output_util.set_allow_interactive(False)
    if redirect_command_output is None:
        cmd_output_util.set_output_redirected(False)
    else:
        cmd_output_util.set_output_redirected(redirect_command_output)

    def handle_yaml_error(e):
        cli_logger.error('Cluster config invalid')
        cli_logger.newline()
        cli_logger.error('Failed to load YAML file ' + cf.bold('{}'), config_file)
        cli_logger.newline()
        with cli_logger.verbatim_error_ctx('PyYAML error:'):
            cli_logger.error(e)
        cli_logger.abort()
    try:
        config = yaml.safe_load(open(config_file).read())
    except FileNotFoundError:
        cli_logger.abort('Provided cluster configuration file ({}) does not exist', cf.bold(config_file))
    except yaml.parser.ParserError as e:
        handle_yaml_error(e)
        raise
    except yaml.scanner.ScannerError as e:
        handle_yaml_error(e)
        raise
    global_event_system.execute_callback(CreateClusterEvent.up_started, {'cluster_config': config})
    importer = _NODE_PROVIDERS.get(config['provider']['type'])
    if not importer:
        cli_logger.abort('Unknown provider type ' + cf.bold('{}') + '\nAvailable providers are: {}', config['provider']['type'], cli_logger.render_list([k for k in _NODE_PROVIDERS.keys() if _NODE_PROVIDERS[k] is not None]))
    printed_overrides = False

    def handle_cli_override(key, override):
        if override is not None:
            if key in config:
                nonlocal printed_overrides
                printed_overrides = True
                cli_logger.warning('`{}` override provided on the command line.\n  Using ' + cf.bold('{}') + cf.dimmed(' [configuration file has ' + cf.bold('{}') + ']'), key, override, config[key])
            config[key] = override
    handle_cli_override('min_workers', override_min_workers)
    handle_cli_override('max_workers', override_max_workers)
    handle_cli_override('cluster_name', override_cluster_name)
    if printed_overrides:
        cli_logger.newline()
    cli_logger.labeled_value('Cluster', config['cluster_name'])
    cli_logger.newline()
    config = _bootstrap_config(config, no_config_cache=no_config_cache)
    try_logging_config(config)
    get_or_create_head_node(config, config_file, no_restart, restart_only, yes, override_cluster_name, no_monitor_on_head)
    return config