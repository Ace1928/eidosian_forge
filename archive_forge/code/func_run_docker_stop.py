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
def run_docker_stop(node, container_name):
    try:
        updater = NodeUpdaterThread(node_id=node, provider_config=config['provider'], provider=provider, auth_config=config['auth'], cluster_name=config['cluster_name'], file_mounts=config['file_mounts'], initialization_commands=[], setup_commands=[], ray_start_commands=[], runtime_hash='', file_mounts_contents_hash='', is_head_node=False, docker_config=config.get('docker'))
        _exec(updater, f'docker stop {container_name}', with_output=False, run_env='host')
    except Exception:
        cli_logger.warning(f'Docker stop failed on {node}')