import argparse
import cProfile
import pstats
import sys
import os
from typing import Dict
import torch
from torch.autograd import profiler
from torch.utils.collect_env import get_env_info
def run_env_analysis():
    print('Running environment analysis...')
    info = get_env_info()
    result: Dict[str, str] = {}
    debug_str = ''
    if info.is_debug_build:
        debug_str = ' DEBUG'
    cuda_avail = ''
    if info.is_cuda_available:
        cuda = info.cuda_runtime_version
        if cuda is not None:
            cuda_avail = 'CUDA ' + cuda
    else:
        cuda = 'CUDA unavailable'
    pip_version = info.pip_version
    pip_list_output = info.pip_packages
    if pip_list_output is None:
        pip_list_output = 'Unable to fetch'
    result = {'debug_str': debug_str, 'pytorch_version': info.torch_version, 'cuda_compiled': compiled_with_cuda(info), 'py_version': f'{sys.version_info[0]}.{sys.version_info[1]}', 'cuda_runtime': cuda_avail, 'pip_version': pip_version, 'pip_list_output': pip_list_output}
    return env_summary.format(**result)