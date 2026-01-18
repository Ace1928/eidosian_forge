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
def get_head_node_ip(config_file: str, override_cluster_name: Optional[str]=None) -> str:
    """Returns head node IP for given configuration file if exists."""
    config = yaml.safe_load(open(config_file).read())
    if override_cluster_name is not None:
        config['cluster_name'] = override_cluster_name
    provider = _get_node_provider(config['provider'], config['cluster_name'])
    head_node = _get_running_head_node(config, config_file, override_cluster_name)
    if config.get('provider', {}).get('use_internal_ips', False):
        head_node_ip = provider.internal_ip(head_node)
    else:
        head_node_ip = provider.external_ip(head_node)
    return head_node_ip