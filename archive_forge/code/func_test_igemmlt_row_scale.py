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
@pytest.mark.parametrize(('dim1', 'dim4', 'inner'), (pytest.param(dim1, dim4, inner, id=f'dim1={dim1!r},dim4={dim4!r},inner={inner!r}') for dim1, dim4, inner in zip(get_test_dims(1, 4 * 1024, n=6), get_test_dims(1, 4 * 1024, n=6), get_test_dims(1, 4 * 1024, n=6))))
@pytest.mark.skip('Row scale has some bugs for ampere')
def test_igemmlt_row_scale(dim1, dim4, inner):
    formatB = F.get_special_format_str()
    err1, err2, err3 = ([], [], [])
    relerr1, relerr2 = ([], [])
    scale = 1
    for i in range(k):
        A = torch.randn(dim1, inner, device='cuda').half()
        B = torch.randn(dim4, inner, device='cuda').half()
        torch.nn.init.xavier_uniform_(B)
        C1 = torch.matmul(A, B.t())
        out1 = torch.matmul(A.half(), B.t().half())
        C1a, C1b, stats1a, stats1b, coo_tensor = F.double_quant(A)
        CB, absmaxB = F.vectorwise_quant(B, quant_type='linear')
        A2, SA = F.nvidia_transform(C1a, 'col32')
        B2, SB = F.nvidia_transform(CB, formatB)
        A1, maxA = F.vectorwise_quant(A, dim=1)
        c = 10.0 * inner * scale
        row_scale = torch.ones_like(maxA) / c
        outC32, SC = F.igemmlt(A2, B2, SA, SB, dtype=torch.int8, row_scale=row_scale)
        C3, S = F.nvidia_transform(outC32, 'row', state=SC)
        maxval = torch.abs(C3).max()
        if maxval == 127:
            scale = 1.5
        else:
            scale = maxval / 120
        out3 = C3 * maxA * absmaxB * c / (127 * 127)
        C4 = torch.matmul(C1a.float(), CB.float().t())
        C2a, C2b, stats2a, stats2b, coo_tensor = F.double_quant(B)
        B2, SB = F.nvidia_transform(C2a, formatB)
        outC32, SC = F.igemmlt(A2, B2, SA, SB)
        out2 = F.mm_dequant(outC32, SC, stats1a, stats2a)
        CA, SA = F.vectorwise_quant(A, dim=1, quant_type='vector')
        CB, SB = F.vectorwise_quant(B, dim=1, quant_type='linear')
        C = torch.matmul(CA.float(), CB.t().float())
        out4 = C * SA * SB / (127 * 127)
        err1.append(torch.abs(out1 - out2).mean().item())
        err2.append(torch.abs(out1 - out3).mean().item())
        err3.append(torch.abs(out1 - out4).mean().item())
    print('')
    print(sum(err1) / len(err1))
    print(sum(err2) / len(err2))
    print(sum(err3) / len(err3))