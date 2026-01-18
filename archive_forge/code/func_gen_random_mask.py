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
def gen_random_mask(self, seq_length):
    num_random_blocks = self.config.num_random_tokens // self.config.block_size
    mask_indices = torch.randint(0, seq_length - 1, size=(seq_length, num_random_blocks))
    random_mask = torch.zeros(seq_length, seq_length).to(dtype=torch.bool)
    random_mask.scatter_(1, mask_indices, True)
    return random_mask