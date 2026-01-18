import argparse
import contextlib
import dataclasses
import enum
import multiprocessing
import os
import random
from collections import deque
from statistics import mean, stdev
from typing import Callable
import torch
def run_one_rank(my_rank, world_size, scenario_name, step, dtype_str, num_rounds, num_warmup_iters, num_bench_iters, profile, conn_from_prev, conn_to_next):
    print(f'RANK {my_rank} started')
    torch.cuda.set_device(my_rank)
    my_device = torch.device(f'cuda:{my_rank}')
    os.environ['RANK'] = f'{my_rank}'
    os.environ['WORLD_SIZE'] = f'{world_size}'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    subgroup = torch.distributed.new_group()
    subgroup_nowait = torch.distributed.new_group()
    subgroup_nowait_nomemcpy = torch.distributed.new_group()
    scenario = SCENARIOS[scenario_name](world_size)
    if step is Step.AllGather:
        M = scenario.num_samples
        N = scenario.inner_dim
        K = scenario.outer_dim
        num_matrices = scenario.num_ag_matrices
    elif step is Step.ReduceScatter:
        M = scenario.num_samples
        N = scenario.outer_dim
        K = scenario.inner_dim
        num_matrices = 1
    dtype = DTYPES[dtype_str]
    scattered_input = torch.randn((M // world_size, K), dtype=dtype, device=my_device)
    gathered_input = torch.randn((M, K), dtype=dtype, device=my_device)
    weights = [torch.randn((K, N), dtype=dtype, device=my_device) for _ in range(num_matrices)]
    gathered_outputs = [torch.randn((M, N), dtype=dtype, device=my_device) for _ in range(num_matrices)]
    scattered_outputs = [torch.randn((M // world_size, N), dtype=dtype, device=my_device) for _ in range(num_matrices)]
    gathered_outputs_nccl_reference = [torch.randn((M, N), dtype=dtype, device=my_device) for _ in range(num_matrices)]
    gathered_outputs_fused = [torch.randn((M, N), dtype=dtype, device=my_device) for _ in range(num_matrices)]
    scattered_outputs_nccl_reference = [torch.randn((M // world_size, N), dtype=dtype, device=my_device) for _ in range(num_matrices)]
    scattered_outputs_fused = [torch.randn((M // world_size, N), dtype=dtype, device=my_device) for _ in range(num_matrices)]

    def run_compute_lower_bound_ag():
        for w, go in zip(weights, gathered_outputs):
            torch.matmul(gathered_input, w, out=go)

    def run_compute_lower_bound_rs():
        for w, go, so in zip(weights, gathered_outputs, scattered_outputs):
            torch.matmul(gathered_input, w, out=go)
            torch.sum(go.view((world_size, M // world_size, N)), dim=0, out=so)

    def run_comms_lower_bound_ag():
        torch.distributed.all_gather_into_tensor(gathered_input, scattered_input)

    def run_comms_lower_bound_rs():
        for so, go in zip(scattered_outputs, gathered_outputs):
            torch.distributed.reduce_scatter_tensor(so, go)

    def run_nccl_reference_ag():
        torch.distributed.all_gather_into_tensor(gathered_input, scattered_input)
        for w, go in zip(weights, gathered_outputs_nccl_reference):
            torch.matmul(gathered_input, w, out=go)

    def run_nccl_reference_rs():
        for w, go, so in zip(weights, gathered_outputs, scattered_outputs_nccl_reference):
            torch.matmul(gathered_input, w, out=go)
            torch.distributed.reduce_scatter_tensor(so, go)

    def run_fused_ag():
        nonlocal gathered_outputs_fused
        from xformers.ops import fused_allgather_and_linear
        gathered_outputs_fused = fused_allgather_and_linear(scattered_input, [w.t() for w in weights], group=subgroup, num_stripes=2, timeout_s=10)

    def run_fused_rs():
        nonlocal scattered_outputs_fused
        from xformers.ops import fused_linear_and_reducescatter
        scattered_outputs_fused = fused_linear_and_reducescatter(gathered_input, [w.t() for w in weights], group=subgroup, num_stripes=2, timeout_s=10)

    def run_fused_nowait_ag():
        nonlocal gathered_outputs_fused
        from xformers.ops import fused_allgather_and_linear
        gathered_outputs_fused = fused_allgather_and_linear(scattered_input, [w.t() for w in weights], group=subgroup_nowait, num_stripes=2, _wait=False, timeout_s=10)

    def run_fused_nowait_rs():
        nonlocal scattered_outputs_fused
        from xformers.ops import fused_linear_and_reducescatter
        scattered_outputs_fused = fused_linear_and_reducescatter(gathered_input, [w.t() for w in weights], group=subgroup_nowait, num_stripes=2, _wait=False, timeout_s=10)

    def run_fused_nowait_nomemcpy_ag():
        nonlocal gathered_outputs_fused
        from xformers.ops import fused_allgather_and_linear
        gathered_outputs_fused = fused_allgather_and_linear(scattered_input, [w.t() for w in weights], group=subgroup_nowait_nomemcpy, num_stripes=2, _wait=False, _memcpy=False, timeout_s=10)

    def run_fused_nowait_nomemcpy_rs():
        nonlocal scattered_outputs_fused
        from xformers.ops import fused_linear_and_reducescatter
        scattered_outputs_fused = fused_linear_and_reducescatter(gathered_input, [w.t() for w in weights], group=subgroup_nowait_nomemcpy, num_stripes=2, _wait=False, _memcpy=False, timeout_s=10)
    print(f'Sizes: ({world_size}x{M // world_size})x({num_matrices}x{N})x{K}')
    if step is Step.AllGather:
        run_nccl_reference_ag()
        run_fused_ag()
        if my_rank == 0:
            print('fused:')
            print('Are equal? ' + ' '.join((str(torch.equal(ref, fus)) for ref, fus in zip(gathered_outputs_nccl_reference, gathered_outputs_fused))))
            print('Are allclose? ' + ' '.join((str(torch.allclose(ref, fus)) for ref, fus in zip(gathered_outputs_nccl_reference, gathered_outputs_fused))))
    elif step is Step.ReduceScatter:
        run_nccl_reference_rs()
        run_fused_rs()
        if my_rank == 0:
            print('fused:')
            print('Are equal? ' + ' '.join((str(torch.equal(ref, fus)) for ref, fus in zip(scattered_outputs_nccl_reference, scattered_outputs_fused))))
            print('Are allclose? ' + ' '.join((str(torch.allclose(ref, fus)) for ref, fus in zip(scattered_outputs_nccl_reference, scattered_outputs_fused))))
    all_benchs = {'compute_lower_bound': Bench(ag=run_compute_lower_bound_ag, rs=run_compute_lower_bound_rs), 'comms_lower_bound': Bench(ag=run_comms_lower_bound_ag, rs=run_comms_lower_bound_rs), 'nccl_reference': Bench(ag=run_nccl_reference_ag, rs=run_nccl_reference_rs), 'fused': Bench(ag=run_fused_ag, rs=run_fused_rs), 'fused_nowait': Bench(ag=run_fused_nowait_ag, rs=run_fused_nowait_rs), 'fused_nowait_nomemcpy': Bench(ag=run_fused_nowait_nomemcpy_ag, rs=run_fused_nowait_nomemcpy_rs)}
    unused_events = deque((tuple((torch.cuda.Event(enable_timing=my_rank == 0) for _ in range(2))) for f in range(len(all_benchs))))
    used_events = deque()
    timings = {}
    gen = random.Random(42)
    if profile:
        profiler = torch.profiler.profile()
    else:
        profiler = contextlib.nullcontext()
    with profiler as p:
        for method in gen.sample(list(all_benchs), k=num_rounds * len(all_benchs), counts=[num_rounds] * len(all_benchs)):
            fun = all_benchs[method][step]
            if unused_events:
                start_ev, end_ev = unused_events.popleft()
            else:
                old_method, start_ev, end_ev = used_events.popleft()
                end_ev.synchronize()
                if my_rank == 0:
                    timings.setdefault(old_method, []).append(start_ev.elapsed_time(end_ev) / num_bench_iters)
            for _ in range(num_warmup_iters):
                fun()
            start_ev.record()
            for _ in range(num_bench_iters):
                fun()
            end_ev.record()
            used_events.append((method, start_ev, end_ev))
        torch.cuda.synchronize()
    if profile:
        p.export_chrome_trace(f'fusion_trace_{my_rank}.json')
    if my_rank == 0:
        for method, start_ev, end_ev in used_events:
            timings.setdefault(method, []).append(start_ev.elapsed_time(end_ev) / num_bench_iters)
        for method in all_benchs:
            print(f'{method} = {mean(timings[method]):g}ms (+/- {stdev(timings[method]):g})')