import itertools
import torch
from torch.utils import benchmark
from triton.ops.matmul import matmul as triton_matmul
from xformers.benchmarks.utils import DTYPE2STR, benchmark_main_helper
from xformers.ops.tiled_matmul import tiled_matmul
def benchmark_tiled_matmul(shape_name, dtype):
    ms, ns, ks = SHAPES[shape_name]
    m, n, k = (sum(ms), sum(ns), sum(ks))
    a = torch.randn((m, k), device='cuda', dtype=dtype)
    b = torch.randn((k, n), device='cuda', dtype=dtype)
    a_tiles = [[y.clone() for y in x.split(ks, dim=1)] for x in a.split(ms, dim=0)]
    b_tiles = [[y.clone() for y in x.split(ns, dim=1)] for x in b.split(ks, dim=0)]
    dtype_str = DTYPE2STR.get(dtype, dtype)
    sub_label = f'{dtype_str} {shape_name} M={'+'.join((f'{m}' for m in ms))} N={'+'.join((f'{n}' for n in ns))} K={'+'.join((f'{k}' for k in ks))}'
    torch.mm(a, b)
    matmul_per_tile(a_tiles, b_tiles)
    triton_matmul(a, b)
    tiled_matmul(a_tiles, b_tiles)
    yield benchmark.Timer(stmt='fn(a, b)', globals={'a': a, 'b': b, 'fn': torch.mm}, label='tiled_matmul', description='pytorch_fused', sub_label=sub_label)
    yield benchmark.Timer(stmt='fn(a, b)', globals={'a': a_tiles, 'b': b_tiles, 'fn': matmul_per_tile}, label='tiled_matmul', description='pytorch_tiled', sub_label=sub_label)
    yield benchmark.Timer(stmt='fn(a, b)', globals={'a': a, 'b': b, 'fn': triton_matmul}, label='tiled_matmul', description='triton_fused', sub_label=sub_label)
    yield benchmark.Timer(stmt='fn(a, b)', globals={'a': a_tiles, 'b': b_tiles, 'fn': tiled_matmul}, label='tiled_matmul', description='xformers_tiled', sub_label=sub_label)