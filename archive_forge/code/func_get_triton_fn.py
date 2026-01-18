import gc
import math
from collections import namedtuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
import triton
from triton.ops.blocksparse import matmul as blocksparse_matmul
from xformers.benchmarks.utils import pretty_barplot
from xformers.components.attention.attention_patterns import (
from xformers.components.attention.core import SparseCS, _matmul_with_mask
def get_triton_fn(self, mask, config, mode='sddmm'):
    if mode == 'sddmm':
        return blocksparse_matmul(layout=mask, block=config.block_size, mode='sdd', device='cuda', trans_a=False, trans_b=True)
    else:
        assert mode == 'spmm'
        return blocksparse_matmul(layout=mask, block=config.block_size, mode='dsd', device='cuda', trans_a=False, trans_b=False)