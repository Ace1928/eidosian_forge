import argparse
import cProfile
import pstats
import sys
import os
from typing import Dict
import torch
from torch.autograd import profiler
from torch.utils.collect_env import get_env_info
def compiled_with_cuda(sysinfo):
    if sysinfo.cuda_compiled_version:
        return f'compiled w/ CUDA {sysinfo.cuda_compiled_version}'
    return 'not compiled w/ CUDA'