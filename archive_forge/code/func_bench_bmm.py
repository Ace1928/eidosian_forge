import itertools
import torch
from torch.utils import benchmark
from xformers.components.attention.core import (
def bench_bmm():
    min_run_time = MIN_RUN_TIME
    prob = 0.9
    device = torch.device('cuda')
    results = []
    for B, M, K in zip(*SHAPES):
        a = torch.rand(B, M, M, device=device)
        a[a < prob] = 0
        b = torch.rand(B, M, K, device=device)
        results.extend([benchmark.Timer(stmt='bmm(a, b)', globals={'a': a, 'b': b, 'bmm': bmm}, label='bmm', sub_label='dense', description=f'B={B}, M={M}, K={K}').blocked_autorange(min_run_time=min_run_time)])
        for sputnik, prob in itertools.product([False, True], SPARSITIES):
            a = _create_random_sparsity(torch.rand(B, M, M, device=device), prob)
            bb = b
            if sputnik:
                a = SparseCS(a, device)
                bb = b
            else:
                a = a.to_sparse()
            results.append(benchmark.Timer(stmt='bmm(a, b)', globals={'a': a, 'b': bb, 'bmm': bmm}, label='bmm', sub_label=f'sparsity {('sputnik' if sputnik else 'pytorch')}: {prob:0.2f}', description=f'B={B}, M={M}, K={K}').blocked_autorange(min_run_time=min_run_time))
    compare = benchmark.Compare(results)
    compare.print()