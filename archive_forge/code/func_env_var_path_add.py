import argparse
import os
import subprocess
import sys
import warnings
from ast import literal_eval
from shutil import which
from typing import Any, Dict, List, Tuple
import torch
from ..commands.config.config_args import SageMakerConfig
from ..utils import (
from ..utils.constants import DEEPSPEED_MULTINODE_LAUNCHERS
from ..utils.other import is_port_in_use, merge_dicts
from .dataclasses import DistributedType, SageMakerDistributedType
def env_var_path_add(env_var_name, path_to_add):
    """
    Extends a path-based environment variable's value with a new path and returns the updated value. It's up to the
    caller to set it in os.environ.
    """
    paths = [p for p in os.environ.get(env_var_name, '').split(':') if len(p) > 0]
    paths.append(str(path_to_add))
    return ':'.join(paths)