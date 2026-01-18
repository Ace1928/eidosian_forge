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
def prepare_simple_launcher_cmd_env(args: argparse.Namespace) -> Tuple[List[str], Dict[str, str]]:
    """
    Prepares and returns the command list and an environment with the correct simple launcher environment variables.
    """
    cmd = []
    if args.no_python and args.module:
        raise ValueError('--module and --no_python cannot be used together')
    if args.mpirun_hostfile is not None:
        mpi_app_name, hostfile_arg, num_proc_arg, proc_per_node_arg = _get_mpirun_args()
        mpirun_ccl = getattr(args, 'mpirun_ccl', None)
        num_machines = args.num_machines
        num_processes = getattr(args, 'num_processes', None)
        nproc_per_node = str(num_processes // num_machines) if num_processes and num_machines else '1'
        cmd += [mpi_app_name, hostfile_arg, args.mpirun_hostfile, proc_per_node_arg, nproc_per_node]
        if num_processes:
            cmd += [num_proc_arg, str(num_processes)]
    if not args.no_python:
        cmd.append(sys.executable)
        if args.module:
            cmd.append('-m')
    cmd.append(args.training_script)
    cmd.extend(args.training_script_args)
    current_env = os.environ.copy()
    current_env['ACCELERATE_USE_CPU'] = str(args.cpu or args.use_cpu)
    if args.debug:
        current_env['ACCELERATE_DEBUG_MODE'] = 'true'
    if args.gpu_ids != 'all' and args.gpu_ids is not None:
        if is_xpu_available():
            current_env['ZE_AFFINITY_MASK'] = args.gpu_ids
        elif is_npu_available():
            current_env['ASCEND_RT_VISIBLE_DEVICES'] = args.gpu_ids
        else:
            current_env['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    if args.num_machines > 1:
        current_env['MASTER_ADDR'] = args.main_process_ip
        current_env['MASTER_PORT'] = str(args.main_process_port)
        if args.mpirun_hostfile is not None:
            current_env['CCL_WORKER_COUNT'] = mpirun_ccl
    elif args.num_processes > 1:
        current_env['MASTER_ADDR'] = args.main_process_ip if args.main_process_ip is not None else '127.0.0.1'
        current_env['MASTER_PORT'] = str(args.main_process_port) if args.main_process_port is not None else '29500'
    try:
        mixed_precision = PrecisionType(args.mixed_precision.lower())
    except ValueError:
        raise ValueError(f'Unknown mixed_precision mode: {args.mixed_precision.lower()}. Choose between {PrecisionType.list()}.')
    current_env['ACCELERATE_MIXED_PRECISION'] = str(mixed_precision)
    try:
        dynamo_backend = DynamoBackend(args.dynamo_backend.upper())
    except ValueError:
        raise ValueError(f'Unknown dynamo backend: {args.dynamo_backend.upper()}. Choose between {DynamoBackend.list()}.')
    current_env['ACCELERATE_DYNAMO_BACKEND'] = dynamo_backend.value
    current_env['ACCELERATE_DYNAMO_MODE'] = args.dynamo_mode
    current_env['ACCELERATE_DYNAMO_USE_FULLGRAPH'] = str(args.dynamo_use_fullgraph)
    current_env['ACCELERATE_DYNAMO_USE_DYNAMIC'] = str(args.dynamo_use_dynamic)
    current_env['OMP_NUM_THREADS'] = str(args.num_cpu_threads_per_process)
    if is_ipex_available():
        current_env['ACCELERATE_USE_IPEX'] = str(args.ipex).lower()
        current_env['ACCELERATE_USE_XPU'] = str(args.use_xpu).lower()
    return (cmd, current_env)