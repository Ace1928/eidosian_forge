import argparse
import cProfile
import pstats
import sys
import os
from typing import Dict
import torch
from torch.autograd import profiler
from torch.utils.collect_env import get_env_info
def print_cprofile_summary(prof, sortby='tottime', topk=15):
    print(cprof_summary)
    cprofile_stats = pstats.Stats(prof).sort_stats(sortby)
    cprofile_stats.print_stats(topk)