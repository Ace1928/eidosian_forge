from typing import Any, Dict
import torch
import triton
from triton.ops.blocksparse import matmul as blocksparse_matmul
from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components.attention.core import SparseCS, _matmul_with_mask
def sparse_step():
    if mode == 'sdd':
        return _matmul_with_mask(a_cs, b_cs, sparse_cs_mask)
    else:
        return sparse_cs_mask.spmm(b_cs)