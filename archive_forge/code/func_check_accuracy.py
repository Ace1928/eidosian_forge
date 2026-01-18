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
def check_accuracy(self, ref_full, res_bsr, mask, config):
    if self.mode == 'sddmm':
        sparse_dot_dsd = blocksparse_matmul(layout=mask, block=config.block_size, mode='dsd', device='cuda', trans_a=False, trans_b=False)
        identity = torch.eye(config.seq_length, config.seq_length, device=device, dtype=self.dtype)
        identity = identity.expand(config.batch_size, config.num_heads, -1, -1)
        res = sparse_dot_dsd(res_bsr, identity)
        full_mask = densify_mask(mask, config)
        ref = ref_full * full_mask.to(dtype=self.dtype, device=device)
        try:
            assert torch.allclose(ref, res, atol=0.001, rtol=0.001)
        except RuntimeError:
            pass
        except AssertionError:
            raise
    else:
        assert self.mode == 'spmm'
        try:
            assert torch.allclose(ref_full, res_bsr, atol=0.001, rtol=0.001)
        except RuntimeError:
            pass
        except AssertionError:
            import pdb
            pdb.set_trace()
            raise