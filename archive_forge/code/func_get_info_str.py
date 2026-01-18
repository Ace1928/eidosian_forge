import dataclasses
import tempfile
from collections import defaultdict
import torch
from torch.autograd import DeviceType
from .utils import create_bandwidth_info_str, do_bench, get_num_bytes
def get_info_str(ms, n_regs, n_spills, shared, prefix=''):
    if not any((x is None for x in [n_regs, n_spills, shared])):
        kernel_detail_str = f'  {n_regs:3} regs  {n_spills:3} spills  {shared:8} shared mem'
    else:
        kernel_detail_str = ''
    gb_per_s = num_gb / (ms / 1000.0)
    return create_bandwidth_info_str(ms, num_gb, gb_per_s, prefix=prefix, suffix=kernel_detail_str)