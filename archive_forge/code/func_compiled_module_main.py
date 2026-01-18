import dataclasses
import tempfile
from collections import defaultdict
import torch
from torch.autograd import DeviceType
from .utils import create_bandwidth_info_str, do_bench, get_num_bytes
def compiled_module_main(benchmark_name, benchmark_compiled_module_fn):
    """
    This is the function called in __main__ block of a compiled module.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark-kernels', '-k', action='store_true', help='Whether to benchmark each individual kernels')
    parser.add_argument('--benchmark-all-configs', '-c', action='store_true', help='Whether to benchmark each individual config for a kernel')
    parser.add_argument('--profile', '-p', action='store_true', help='Whether to profile the compiled module')
    args = parser.parse_args()
    if args.benchmark_kernels:
        benchmark_all_kernels(benchmark_name, args.benchmark_all_configs)
    else:
        times = 10
        repeat = 10
        wall_time_ms = benchmark_compiled_module_fn(times=times, repeat=repeat) / times * 1000
        if not args.profile:
            return
        with torch.profiler.profile(record_shapes=True) as p:
            benchmark_compiled_module_fn(times=times, repeat=repeat)
        path = f'{tempfile.gettempdir()}/compiled_module_profile.json'
        p.export_chrome_trace(path)
        print(f'Profiling result for a compiled module of benchmark {benchmark_name}:')
        print(f'Chrome trace for the profile is written to {path}')
        event_list = p.key_averages(group_by_input_shape=True)
        print(event_list.table(sort_by='self_cuda_time_total', row_limit=10))
        parse_profile_event_list(benchmark_name, event_list, wall_time_ms, times * repeat)