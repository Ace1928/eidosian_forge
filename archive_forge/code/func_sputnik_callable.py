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
def sputnik_callable(self, a, b, mask, config):
    assert self.mode == 'sddmm'
    a_cs, b_cs, sparse_mask_cs = self.prepare_sputnik_inputs(a, b, config, mask)

    def sputnik_fn():
        return _matmul_with_mask(a_cs, b_cs, sparse_mask_cs)
    return sputnik_fn