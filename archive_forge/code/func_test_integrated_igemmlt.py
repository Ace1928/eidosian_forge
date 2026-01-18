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
@pytest.mark.parametrize(('dim1', 'dim4', 'inner'), (pytest.param(dim1, dim4, inner, id=f'dim1={dim1!r},dim4={dim4!r},inner={inner!r}') for dim1, dim4, inner in zip(get_test_dims(1, 4 * 1024, n=4), get_test_dims(1, 4 * 1024, n=4), get_test_dims(1, 4 * 1024, n=4))))
def test_integrated_igemmlt(dim1, dim4, inner):
    for i in range(k):
        A = torch.randn(dim1, inner, device='cuda').half()
        B = torch.randn(dim4, inner, device='cuda').half()
        out1 = torch.matmul(A.half(), B.t().half())
        C1a, C1b, stats1a, stats1b, coo_tensor = F.double_quant(A)
        C2a, C2b, stats2a, stats2b, coo_tensor = F.double_quant(B)
        A1, maxA = F.vectorwise_quant(A, dim=1)
        B1, maxB = F.vectorwise_quant(B, dim=1)
        torch.testing.assert_close(maxA.flatten().float(), stats1a)
        torch.testing.assert_close(maxB.flatten().float(), stats2a)
        torch.testing.assert_close(C1a, A1, rtol=0, atol=1)
        torch.testing.assert_close(C2a, B1, rtol=0, atol=1)
        A2, SA = F.nvidia_transform(C1a, 'col32')
        B2, SB = F.nvidia_transform(C2a, 'col_turing')
        outC32, SC = F.igemmlt(A2, B2, SA, SB)
        out2 = F.mm_dequant(outC32, SC, stats1a, stats2a)
        A2, SA = F.nvidia_transform(A1, 'col32')
        B2, SB = F.nvidia_transform(B1, 'col_turing')
        C2, SC = F.igemmlt(A2, B2, SA, SB)
        C3, S = F.nvidia_transform(C2, 'row', state=SC)
        out3 = F.vectorwise_mm_dequant(C3.float(), maxA, maxB.t())
        err1 = torch.abs(out1 - out2).mean().item()
        err2 = torch.abs(out1 - out3).mean().item()
        assert err2 <= err1 * 1.025