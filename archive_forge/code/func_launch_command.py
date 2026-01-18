import argparse
import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path
import psutil
import torch
from accelerate.commands.config import default_config_file, load_config_from_file
from accelerate.commands.config.config_args import SageMakerConfig
from accelerate.commands.config.config_utils import DYNAMO_BACKENDS
from accelerate.commands.utils import CustomArgumentParser
from accelerate.state import get_int_from_env
from accelerate.utils import (
from accelerate.utils.constants import DEEPSPEED_MULTINODE_LAUNCHERS, TORCH_DYNAMO_MODES
def launch_command(args):
    args, defaults, mp_from_config_flag = _validate_launch_command(args)
    if args.use_deepspeed and (not args.cpu):
        args.deepspeed_fields_from_accelerate_config = list(defaults.deepspeed_config.keys()) if defaults else []
        if mp_from_config_flag:
            args.deepspeed_fields_from_accelerate_config.append('mixed_precision')
        args.deepspeed_fields_from_accelerate_config = ','.join(args.deepspeed_fields_from_accelerate_config)
        deepspeed_launcher(args)
    elif args.use_fsdp and (not args.cpu):
        multi_gpu_launcher(args)
    elif args.use_megatron_lm and (not args.cpu):
        multi_gpu_launcher(args)
    elif args.multi_gpu and (not args.cpu):
        multi_gpu_launcher(args)
    elif args.tpu and (not args.cpu):
        if args.tpu_use_cluster:
            tpu_pod_launcher(args)
        else:
            tpu_launcher(args)
    elif defaults is not None and defaults.compute_environment == ComputeEnvironment.AMAZON_SAGEMAKER:
        sagemaker_launcher(defaults, args)
    else:
        simple_launcher(args)