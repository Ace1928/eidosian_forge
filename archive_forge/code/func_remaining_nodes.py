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
def remaining_nodes():
    workers = provider.non_terminated_nodes({TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
    if keep_min_workers:
        min_workers = config.get('min_workers', 0)
        cli_logger.print('{} random worker nodes will not be shut down. ' + cf.dimmed('(due to {})'), cf.bold(min_workers), cf.bold('--keep-min-workers'))
        workers = random.sample(workers, len(workers) - min_workers)
    if workers_only:
        cli_logger.print('The head node will not be shut down. ' + cf.dimmed('(due to {})'), cf.bold('--workers-only'))
        return workers
    head = provider.non_terminated_nodes({TAG_RAY_NODE_KIND: NODE_KIND_HEAD})
    return head + workers