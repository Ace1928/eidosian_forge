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
def prepare_tpu(args: argparse.Namespace, current_env: Dict[str, str], pod: bool=False) -> Tuple[argparse.Namespace, Dict[str, str]]:
    """
    Prepares and returns an environment with the correct TPU environment variables.
    """
    if args.mixed_precision == 'bf16' and is_torch_xla_available(check_is_tpu=True):
        if args.downcast_bf16:
            current_env['XLA_DOWNCAST_BF16'] = '1'
        else:
            current_env['XLA_USE_BF16'] = '1'
    if args.debug:
        current_env['ACCELERATE_DEBUG_MODE'] = 'true'
    if pod:
        args.vm = args.tpu_vm
        args.tpu = args.tpu_name
    return (args, current_env)