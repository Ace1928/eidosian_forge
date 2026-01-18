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
def plot_mask(mask, config, filename):
    sparsity = get_sparsity(mask)
    batch_size = config.batch_size
    num_heads = config.num_heads
    seq_len = config.seq_length
    proxy = torch.ones(batch_size, num_heads, seq_len, seq_len, dtype=torch.bool)
    proxy = triton.testing.mask_tensor(proxy, mask, config.block_size, False)
    proxy = proxy[0][0]
    f = plt.figure()
    plt.imshow(proxy.logical_not(), cmap='gray')
    plt.suptitle('Sparsity = ' + str(sparsity) + '%')
    plt.savefig(filename)
    plt.close(f)