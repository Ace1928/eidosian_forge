from itertools import product
import math
import random
import time
import einops
import numpy as np
import pytest
from scipy.stats import norm
import torch
import bitsandbytes as bnb
from bitsandbytes import functional as F
from tests.helpers import (
@pytest.mark.parametrize('dim1', get_test_dims(64, 256, n=2), ids=id_formatter('dim1'))
@pytest.mark.parametrize('dim4', get_test_dims(64, 1024, n=2), ids=id_formatter('dim4'))
@pytest.mark.parametrize('dims', (2,), ids=id_formatter('dims'))
@pytest.mark.parametrize('formatB', ['col_turing', 'col_ampere'], ids=id_formatter('formatB'))
@pytest.mark.parametrize('has_bias', TRUE_FALSE, ids=id_formatter('has_bias'))
def test_dequant_mm(dim1, dim4, dims, formatB, has_bias):
    inner = torch.randint(1, 128, size=(1,)).item()
    bias = None
    if has_bias:
        bias = torch.randn(dim4, device='cuda', dtype=torch.float16)
    formatB = F.get_special_format_str()
    for i in range(1):
        A = torch.randn(dim1, inner, device='cuda')
        B = torch.randn(dim4, inner, device='cuda')
        C1 = torch.matmul(A.half(), B.t().half())
        if has_bias:
            C1 += bias
        A1, maxA = F.vectorwise_quant(A, dim=1)
        B1, maxB = F.vectorwise_quant(B, dim=1)
        A2, SA = F.nvidia_transform(A1, 'col32')
        B2, SB = F.nvidia_transform(B1, formatB)
        C2, SC = F.igemmlt(A2, B2, SA, SB)
        C3, S = F.nvidia_transform(C2, 'row', state=SC)
        C4 = F.vectorwise_mm_dequant(C3.float(), maxA, maxB.t())
        if has_bias:
            C4 += bias
        std = C1.std(0).view(1, -1)
        C1 /= std
        C4 /= std
        C5 = F.mm_dequant(C2, SC, maxA.flatten(), maxB.flatten(), bias=bias)
        n = C5.numel()
        assert_all_approx_close(C1, C4, atol=0.015, rtol=0.1, count=int(0.01 * n))