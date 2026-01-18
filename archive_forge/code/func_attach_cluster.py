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
def attach_cluster(config_file: str, start: bool, use_screen: bool, use_tmux: bool, override_cluster_name: Optional[str], no_config_cache: bool=False, new: bool=False, port_forward: Optional[Port_forward]=None) -> None:
    """Attaches to a screen for the specified cluster.

    Arguments:
        config_file: path to the cluster yaml
        start: whether to start the cluster if it isn't up
        use_screen: whether to use screen as multiplexer
        use_tmux: whether to use tmux as multiplexer
        override_cluster_name: set the name of the cluster
        new: whether to force a new screen
        port_forward ( (int,int) or list[(int,int)] ): port(s) to forward
    """
    if use_tmux:
        if new:
            cmd = 'tmux new'
        else:
            cmd = 'tmux attach || tmux new'
    elif use_screen:
        if new:
            cmd = 'screen -L'
        else:
            cmd = 'screen -L -xRR'
    else:
        if new:
            raise ValueError('--new only makes sense if passing --screen or --tmux')
        cmd = '$SHELL'
    exec_cluster(config_file, cmd=cmd, run_env='auto', screen=False, tmux=False, stop=False, start=start, override_cluster_name=override_cluster_name, no_config_cache=no_config_cache, port_forward=port_forward, _allow_uninitialized_state=True)