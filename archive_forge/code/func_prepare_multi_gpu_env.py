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
def prepare_multi_gpu_env(args: argparse.Namespace) -> Dict[str, str]:
    """
    Prepares and returns an environment with the correct multi-GPU environment variables.
    """
    num_processes = args.num_processes
    num_machines = args.num_machines
    main_process_ip = args.main_process_ip
    main_process_port = args.main_process_port
    if num_machines > 1:
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
    mixed_precision = args.mixed_precision.lower()
    try:
        mixed_precision = PrecisionType(mixed_precision)
    except ValueError:
        raise ValueError(f'Unknown mixed_precision mode: {mixed_precision}. Choose between {PrecisionType.list()}.')
    current_env['ACCELERATE_MIXED_PRECISION'] = str(mixed_precision)
    try:
        dynamo_backend = DynamoBackend(args.dynamo_backend.upper())
    except ValueError:
        raise ValueError(f'Unknown dynamo backend: {args.dynamo_backend.upper()}. Choose between {DynamoBackend.list()}.')
    current_env['ACCELERATE_DYNAMO_BACKEND'] = dynamo_backend.value
    current_env['ACCELERATE_DYNAMO_MODE'] = args.dynamo_mode
    current_env['ACCELERATE_DYNAMO_USE_FULLGRAPH'] = str(args.dynamo_use_fullgraph)
    current_env['ACCELERATE_DYNAMO_USE_DYNAMIC'] = str(args.dynamo_use_dynamic)
    if args.use_fsdp:
        current_env['ACCELERATE_USE_FSDP'] = 'true'
        if args.fsdp_cpu_ram_efficient_loading and (not args.fsdp_sync_module_states):
            raise ValueError('When using `--fsdp_cpu_ram_efficient_loading` set `--fsdp_sync_module_states` to `True`')
        current_env['FSDP_SHARDING_STRATEGY'] = str(args.fsdp_sharding_strategy)
        current_env['FSDP_OFFLOAD_PARAMS'] = str(args.fsdp_offload_params).lower()
        current_env['FSDP_MIN_NUM_PARAMS'] = str(args.fsdp_min_num_params)
        if args.fsdp_auto_wrap_policy is not None:
            current_env['FSDP_AUTO_WRAP_POLICY'] = str(args.fsdp_auto_wrap_policy)
        if args.fsdp_transformer_layer_cls_to_wrap is not None:
            current_env['FSDP_TRANSFORMER_CLS_TO_WRAP'] = str(args.fsdp_transformer_layer_cls_to_wrap)
        if args.fsdp_backward_prefetch_policy is not None:
            warnings.warn('`fsdp_backward_prefetch_policy` is deprecated and will be removed in version 0.27.0 of 🤗 Accelerate. Use `fsdp_backward_prefetch` instead', FutureWarning)
            args.fsdp_backward_prefetch = args.fsdp_backward_prefetch_policy
        if args.fsdp_backward_prefetch is not None:
            current_env['FSDP_BACKWARD_PREFETCH'] = str(args.fsdp_backward_prefetch)
        if args.fsdp_state_dict_type is not None:
            current_env['FSDP_STATE_DICT_TYPE'] = str(args.fsdp_state_dict_type)
        current_env['FSDP_FORWARD_PREFETCH'] = str(args.fsdp_forward_prefetch).lower()
        current_env['FSDP_USE_ORIG_PARAMS'] = str(args.fsdp_use_orig_params).lower()
        current_env['FSDP_CPU_RAM_EFFICIENT_LOADING'] = str(args.fsdp_cpu_ram_efficient_loading).lower()
        current_env['FSDP_SYNC_MODULE_STATES'] = str(args.fsdp_sync_module_states).lower()
    if args.use_megatron_lm:
        prefix = 'MEGATRON_LM_'
        current_env['ACCELERATE_USE_MEGATRON_LM'] = 'true'
        current_env[prefix + 'TP_DEGREE'] = str(args.megatron_lm_tp_degree)
        current_env[prefix + 'PP_DEGREE'] = str(args.megatron_lm_pp_degree)
        current_env[prefix + 'GRADIENT_CLIPPING'] = str(args.megatron_lm_gradient_clipping)
        if args.megatron_lm_num_micro_batches is not None:
            current_env[prefix + 'NUM_MICRO_BATCHES'] = str(args.megatron_lm_num_micro_batches)
        if args.megatron_lm_sequence_parallelism is not None:
            current_env[prefix + 'SEQUENCE_PARALLELISM'] = str(args.megatron_lm_sequence_parallelism)
        if args.megatron_lm_recompute_activations is not None:
            current_env[prefix + 'RECOMPUTE_ACTIVATIONS'] = str(args.megatron_lm_recompute_activations)
        if args.megatron_lm_use_distributed_optimizer is not None:
            current_env[prefix + 'USE_DISTRIBUTED_OPTIMIZER'] = str(args.megatron_lm_use_distributed_optimizer)
    current_env['OMP_NUM_THREADS'] = str(args.num_cpu_threads_per_process)
    return current_env