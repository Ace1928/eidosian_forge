import dataclasses
import tempfile
from collections import defaultdict
import torch
from torch.autograd import DeviceType
from .utils import create_bandwidth_info_str, do_bench, get_num_bytes
def report_category(category, profile_events):
    from tabulate import tabulate
    profile_events.sort(key=lambda ev: ev.self_cuda_time_ms, reverse=True)
    rows = []
    total_time = 0.0
    print(f'\n  == {category} category kernels == ')
    for ev in profile_events:
        total_time += ev.self_cuda_time_ms
        percent = f'{ev.self_cuda_time_ms / wall_time_ms * 100:.2f}%'
        rows.append([ev.key[:120], ev.self_cuda_time_ms, ev.count, percent])
    rows.append(['Total', total_time, '', f'{total_time / wall_time_ms * 100:.2f}%'])
    print(tabulate(rows, headers=['Kernel', 'Self CUDA TIME (ms)', 'Count', 'Percent']))
    return total_time