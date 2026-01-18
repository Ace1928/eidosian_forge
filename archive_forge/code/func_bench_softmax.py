import itertools
import torch
from torch.utils import benchmark
from xformers.components.attention.core import (
def bench_softmax():
    min_run_time = MIN_RUN_TIME
    prob = 0.9
    device = torch.device('cuda')
    results = []
    for B, M, K in zip(*SHAPES):
        a = torch.rand(B, M, M, device=device)
        a[a < prob] = 0
        results.extend([benchmark.Timer(stmt='_softmax(a)', globals={'a': a, '_softmax': _softmax}, label='softmax', sub_label='dense', description=f'B={B}, M={M}, K={K}').blocked_autorange(min_run_time=min_run_time)])
        for sputnik, prob in itertools.product([False, True], SPARSITIES):
            a = _create_random_sparsity(torch.rand(B, M, M, device=device), prob)
            if sputnik:
                a = SparseCS(a, device)
            else:
                a = a.to_sparse()
            results.append(benchmark.Timer(stmt='_softmax(a)', globals={'a': a, '_softmax': _softmax}, label='softmax', sub_label=f'sparsity {('sputnik' if sputnik else 'pytorch')}: {prob:0.2f}', description=f'B={B}, M={M}, K={K}').blocked_autorange(min_run_time=min_run_time))
    compare = benchmark.Compare(results)
    compare.print()