import dataclasses
import tempfile
from collections import defaultdict
import torch
from torch.autograd import DeviceType
from .utils import create_bandwidth_info_str, do_bench, get_num_bytes
def parse_profile_event_list(benchmark_name, event_list, wall_time_ms, nruns):

    def get_self_cuda_time(ev):
        """
        ev.self_cuda_time_total is in microsecond. Convert to millisecond.
        """
        return ev.self_cuda_time_total / 1000 / nruns
    all_events = defaultdict(list)

    def add_event(ev, category):
        profile_ev = ProfileEvent(category=category, key=ev.key, self_cuda_time_ms=get_self_cuda_time(ev), count=ev.count / nruns)
        all_events[category].append(profile_ev)
    for ev in event_list:
        assert not ev.is_legacy, "Don't support the legacy profiler"
        if ev.device_type == DeviceType.CPU:
            continue
        category = 'unknown'
        if ev.key.startswith('triton_'):
            if ev.key.startswith('triton_poi'):
                category = 'triton_pointwise'
            elif ev.key.startswith('triton_red'):
                category = 'triton_reduction'
            elif ev.key.startswith('triton_per'):
                category = 'triton_persistent_reduction'
            else:
                category = 'triton_unknown'
        add_event(ev, category)

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

    def report():
        category_list = ['triton_pointwise', 'triton_reduction', 'triton_persistent_reduction', 'triton_unknown', 'unknown']
        assert set(all_events.keys()).issubset(set(category_list)), f'{list(all_events.keys())}'
        per_category_wall_time = {}
        total_cuda_ms = 0.0
        for category in category_list:
            if category in all_events:
                _time = report_category(category, all_events[category])
                per_category_wall_time[category] = _time
                total_cuda_ms += _time
        gpu_busy_percent = f'{total_cuda_ms / wall_time_ms * 100:.2f}%'
        print(f'\nPercent of time when GPU is busy: {gpu_busy_percent}')
        print(f'Total wall time {wall_time_ms:.3f} ms')
        tabulate_line = f'Output for tabulate: {benchmark_name}'
        for category in category_list:
            percent = f'{per_category_wall_time.get(category, 0.0) / wall_time_ms * 100:.2f}%'
            tabulate_line += f', {percent}'
        tabulate_line += f', {gpu_busy_percent}, {wall_time_ms:.3f}ms'
        print(tabulate_line)
    report()