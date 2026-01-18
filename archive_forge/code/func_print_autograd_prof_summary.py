import argparse
import cProfile
import pstats
import sys
import os
from typing import Dict
import torch
from torch.autograd import profiler
from torch.utils.collect_env import get_env_info
def print_autograd_prof_summary(prof, mode, sortby='cpu_time', topk=15):
    valid_sortby = ['cpu_time', 'cuda_time', 'cpu_time_total', 'cuda_time_total', 'count']
    if sortby not in valid_sortby:
        warn = 'WARNING: invalid sorting option for autograd profiler results: {}\nExpected `cpu_time`, `cpu_time_total`, or `count`. Defaulting to `cpu_time`.'
        print(warn.format(sortby))
        sortby = 'cpu_time'
    if mode == 'CUDA':
        cuda_warning = '\n\tBecause the autograd profiler uses the CUDA event API,\n\tthe CUDA time column reports approximately max(cuda_time, cpu_time).\n\tPlease ignore this output if your code does not use CUDA.\n'
    else:
        cuda_warning = ''
    sorted_events = sorted(prof.function_events, key=lambda x: getattr(x, sortby), reverse=True)
    topk_events = sorted_events[:topk]
    result = {'mode': mode, 'description': f'top {topk} events sorted by {sortby}', 'output': torch.autograd.profiler_util._build_table(topk_events), 'cuda_warning': cuda_warning}
    print(autograd_prof_summary.format(**result))