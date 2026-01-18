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
def prepare_deepspeed_cmd_env(args: argparse.Namespace) -> Tuple[List[str], Dict[str, str]]:
    """
    Prepares and returns the command list and an environment with the correct DeepSpeed environment variables.
    """
    num_processes = args.num_processes
    num_machines = args.num_machines
    main_process_ip = args.main_process_ip
    main_process_port = args.main_process_port
    cmd = None
    if args.deepspeed_multinode_launcher is None:
        args.deepspeed_multinode_launcher = DEEPSPEED_MULTINODE_LAUNCHERS[0]
    if num_machines > 1 and args.deepspeed_multinode_launcher != DEEPSPEED_MULTINODE_LAUNCHERS[1]:
        cmd = ['deepspeed', '--no_local_rank']
        cmd.extend(['--hostfile', str(args.deepspeed_hostfile), '--launcher', str(args.deepspeed_multinode_launcher)])
        if args.deepspeed_exclusion_filter is not None:
            cmd.extend(['--exclude', str(args.deepspeed_exclusion_filter)])
        elif args.deepspeed_inclusion_filter is not None:
            cmd.extend(['--include', str(args.deepspeed_inclusion_filter)])
        else:
            cmd.extend(['--num_gpus', str(args.num_processes // args.num_machines)])
        cmd.extend(['--master_port', str(main_process_port)])
        if args.module and args.no_python:
            raise ValueError('--module and --no_python cannot be used together')
        elif args.module:
            cmd.append('--module')
        elif args.no_python:
            cmd.append('--no_python')
        cmd.append(args.training_script)
        cmd.extend(args.training_script_args)
    elif num_machines > 1 and args.deepspeed_multinode_launcher == DEEPSPEED_MULTINODE_LAUNCHERS[1]:
        args.nproc_per_node = str(num_processes // num_machines)
        args.nnodes = str(num_machines)
        args.node_rank = int(args.machine_rank)
        if getattr(args, 'same_network', False):
            args.master_addr = str(main_process_ip)
            args.master_port = str(main_process_port)
        else:
            args.rdzv_endpoint = f'{main_process_ip}:{main_process_port}'
    else:
        args.nproc_per_node = str(num_processes)
        if main_process_port is not None:
            args.master_port = str(main_process_port)
    if main_process_port is None:
        main_process_port = 29500
    need_port_check = num_machines <= 1 or int(args.machine_rank) == 0
    if need_port_check and is_port_in_use(main_process_port):
        raise ConnectionError(f'Tried to launch distributed communication on port `{main_process_port}`, but another process is utilizing it. Please specify a different port (such as using the `--main_process_port` flag or specifying a different `main_process_port` in your config file) and rerun your script. To automatically use the next open port (on a single node), you can set this to `0`.')
    if args.module and args.no_python:
        raise ValueError('--module and --no_python cannot be used together')
    elif args.module:
        args.module = True
    elif args.no_python:
        args.no_python = True
    current_env = os.environ.copy()
    if args.debug:
        current_env['ACCELERATE_DEBUG_MODE'] = 'true'
    gpu_ids = getattr(args, 'gpu_ids', 'all')
    if gpu_ids != 'all' and args.gpu_ids is not None:
        if is_xpu_available():
            current_env['ZE_AFFINITY_MASK'] = gpu_ids
        elif is_npu_available():
            current_env['ASCEND_RT_VISIBLE_DEVICES'] = gpu_ids
        else:
            current_env['CUDA_VISIBLE_DEVICES'] = gpu_ids
    try:
        mixed_precision = PrecisionType(args.mixed_precision.lower())
    except ValueError:
        raise ValueError(f'Unknown mixed_precision mode: {args.mixed_precision.lower()}. Choose between {PrecisionType.list()}.')
    current_env['PYTHONPATH'] = env_var_path_add('PYTHONPATH', os.path.abspath('.'))
    current_env['ACCELERATE_MIXED_PRECISION'] = str(mixed_precision)
    current_env['ACCELERATE_CONFIG_DS_FIELDS'] = str(args.deepspeed_fields_from_accelerate_config).lower()
    current_env['ACCELERATE_USE_DEEPSPEED'] = 'true'
    if args.zero_stage is not None:
        current_env['ACCELERATE_DEEPSPEED_ZERO_STAGE'] = str(args.zero_stage)
    if args.gradient_accumulation_steps is not None:
        current_env['ACCELERATE_GRADIENT_ACCUMULATION_STEPS'] = str(args.gradient_accumulation_steps)
    if args.gradient_clipping is not None:
        current_env['ACCELERATE_GRADIENT_CLIPPING'] = str(args.gradient_clipping).lower()
    if args.offload_optimizer_device is not None:
        current_env['ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE'] = str(args.offload_optimizer_device).lower()
    if args.offload_param_device is not None:
        current_env['ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE'] = str(args.offload_param_device).lower()
    if args.zero3_init_flag is not None:
        current_env['ACCELERATE_DEEPSPEED_ZERO3_INIT'] = str(args.zero3_init_flag).lower()
    if args.zero3_save_16bit_model is not None:
        current_env['ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL'] = str(args.zero3_save_16bit_model).lower()
    if args.deepspeed_config_file is not None:
        current_env['ACCELERATE_DEEPSPEED_CONFIG_FILE'] = str(args.deepspeed_config_file)
    return (cmd, current_env)