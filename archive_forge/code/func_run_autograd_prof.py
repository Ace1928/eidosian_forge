import argparse
import cProfile
import pstats
import sys
import os
from typing import Dict
import torch
from torch.autograd import profiler
from torch.utils.collect_env import get_env_info
def run_autograd_prof(code, globs):

    def run_prof(use_cuda=False):
        with profiler.profile(use_cuda=use_cuda) as prof:
            exec(code, globs, None)
        return prof
    print('Running your script with the autograd profiler...')
    result = [run_prof(use_cuda=False)]
    if torch.cuda.is_available():
        result.append(run_prof(use_cuda=True))
    else:
        result.append(None)
    return result