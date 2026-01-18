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