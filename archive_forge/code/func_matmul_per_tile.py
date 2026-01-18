import itertools
import torch
from torch.utils import benchmark
from triton.ops.matmul import matmul as triton_matmul
from xformers.benchmarks.utils import DTYPE2STR, benchmark_main_helper
from xformers.ops.tiled_matmul import tiled_matmul
def matmul_per_tile(a, b):
    c = []
    for n in range(len(a)):
        c.append([])
        for m in range(len(b[0])):
            c[-1].append(sum([torch.matmul(a[n][k], b[k][m]) for k in range(len(a[0]))]))
    return c