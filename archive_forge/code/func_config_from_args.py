import logging
import os
import sys
import uuid
from argparse import REMAINDER, ArgumentParser
from typing import Callable, List, Tuple, Union
import torch
from torch.distributed.argparse_util import check_env, env
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.elastic.rendezvous.utils import _parse_rendezvous_config
from torch.distributed.elastic.utils import macros
from torch.distributed.elastic.utils.logging import get_logger
from torch.distributed.launcher.api import LaunchConfig, elastic_launch
from torch.utils.backend_registration import _get_custom_mod_func
def config_from_args(args) -> Tuple[LaunchConfig, Union[Callable, str], List[str]]:
    min_nodes, max_nodes = parse_min_max_nnodes(args.nnodes)
    assert 0 < min_nodes <= max_nodes
    assert args.max_restarts >= 0
    if hasattr(args, 'master_addr') and args.rdzv_backend != 'static' and (not args.rdzv_endpoint):
        log.warning('master_addr is only used for static rdzv_backend and when rdzv_endpoint is not specified.')
    nproc_per_node = determine_local_world_size(args.nproc_per_node)
    if 'OMP_NUM_THREADS' not in os.environ and nproc_per_node > 1:
        omp_num_threads = 1
        log.warning('\n*****************************************\nSetting OMP_NUM_THREADS environment variable for each process to be %s in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. \n*****************************************', omp_num_threads)
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    log_line_prefix_template = os.getenv('TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE')
    rdzv_configs = _parse_rendezvous_config(args.rdzv_conf)
    if args.rdzv_backend == 'static':
        rdzv_configs['rank'] = args.node_rank
    rdzv_endpoint = get_rdzv_endpoint(args)
    config = LaunchConfig(min_nodes=min_nodes, max_nodes=max_nodes, nproc_per_node=nproc_per_node, run_id=args.rdzv_id, role=args.role, rdzv_endpoint=rdzv_endpoint, rdzv_backend=args.rdzv_backend, rdzv_configs=rdzv_configs, max_restarts=args.max_restarts, monitor_interval=args.monitor_interval, start_method=args.start_method, redirects=Std.from_str(args.redirects), tee=Std.from_str(args.tee), log_dir=args.log_dir, log_line_prefix_template=log_line_prefix_template, local_addr=args.local_addr)
    with_python = not args.no_python
    cmd: Union[Callable, str]
    cmd_args = []
    use_env = get_use_env(args)
    if args.run_path:
        cmd = run_script_path
        cmd_args.append(args.training_script)
    elif with_python:
        cmd = os.getenv('PYTHON_EXEC', sys.executable)
        cmd_args.append('-u')
        if args.module:
            cmd_args.append('-m')
        cmd_args.append(args.training_script)
    else:
        if args.module:
            raise ValueError("Don't use both the '--no-python' flag and the '--module' flag at the same time.")
        cmd = args.training_script
    if not use_env:
        cmd_args.append(f'--local-rank={macros.local_rank}')
    cmd_args.extend(args.training_script_args)
    return (config, cmd, cmd_args)