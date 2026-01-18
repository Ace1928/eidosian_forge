import itertools
import torch
from torch.utils import benchmark
from xformers.components.attention.core import (
def bench_matmul_with_mask():
    min_run_time = MIN_RUN_TIME
    prob = 0.9
    device = torch.device('cuda')
    results = []
    for B, M, K in zip(*SHAPES):
        a = torch.rand(B, M, K, device=device)
        b = torch.rand(B, K, M, device=device)
        mask = torch.rand(B, M, M, device=device) > prob
        results.extend([benchmark.Timer(stmt='_matmul_with_mask(a, b, mask)', globals={'a': a, 'b': b, 'mask': None, '_matmul_with_mask': _matmul_with_mask}, label='matmul_with_mask', sub_label='dense', description=f'B={B}, M={M}, K={K}').blocked_autorange(min_run_time=min_run_time), benchmark.Timer(stmt='_matmul_with_mask(a, b, mask)', globals={'a': a, 'b': b, 'mask': mask, '_matmul_with_mask': _matmul_with_mask}, label='matmul_with_mask', sub_label='dense with masking', description=f'B={B}, M={M}, K={K}').blocked_autorange(min_run_time=min_run_time)])
        for sputnik, prob in itertools.product([False, True], SPARSITIES):
            mask = _create_random_sparsity(torch.ones(B, M, M, dtype=torch.bool, device=device), prob)
            aa = a
            bb = b
            if sputnik:
                mask = SparseCS(mask, device)
                aa = a
                bb = b.transpose(-2, -1).contiguous().transpose(-2, -1)
            else:
                mask = mask.to_sparse()
            results.append(benchmark.Timer(stmt='_matmul_with_mask(a, b, mask)', globals={'a': aa, 'b': bb, 'mask': mask, '_matmul_with_mask': _matmul_with_mask}, label='matmul_with_mask', sub_label=f'sparsity {('sputnik' if sputnik else 'pytorch')}: {prob:0.2f}', description=f'B={B}, M={M}, K={K}').blocked_autorange(min_run_time=min_run_time))
    compare = benchmark.Compare(results)
    compare.print()