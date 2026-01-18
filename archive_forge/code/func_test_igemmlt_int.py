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
@pytest.mark.parametrize('dim1', get_test_dims(1, 256, n=1), ids=id_formatter('dim1'))
@pytest.mark.parametrize('dim2', get_test_dims(32, 512, n=1), ids=id_formatter('dim2'))
@pytest.mark.parametrize('dim3', get_test_dims(32, 1024, n=1), ids=id_formatter('dim3'))
@pytest.mark.parametrize('dim4', get_test_dims(32, 1024, n=1), ids=id_formatter('dim4'))
@pytest.mark.parametrize('dims', (2, 3), ids=id_formatter('dims'))
@pytest.mark.parametrize('ldb', (0,), ids=id_formatter('ldb'))
def test_igemmlt_int(dim1, dim2, dim3, dim4, dims, ldb):
    for i in range(k):
        if dims == 2:
            A = torch.randint(-128, 127, size=(dim1, dim3), device='cuda').to(torch.int8)
        elif dims == 3:
            A = torch.randint(-128, 127, size=(dim1, dim2, dim3), device='cuda').to(torch.int8)
        B = torch.randint(-128, 127, size=(dim4, dim3), device='cuda').to(torch.int8)
        C1 = torch.matmul(A.float(), B.t().float())
        A2, SA = F.transform(A, 'col32')
        B2, SB = F.transform(B, 'col_turing')
        C2, SC = F.igemmlt(A2, B2, SA, SB)
        C3, S = F.nvidia_transform(C2, 'row', state=SC)
        torch.testing.assert_close(C1, C3.float())
        B = torch.randint(-128, 127, size=(dim3, dim4), device='cuda').to(torch.int8)
        C1 = torch.matmul(A.float(), B.float())
        B2t, SBt = F.transform(B, 'col_turing', transpose=True)
        C2, SC = F.igemmlt(A2, B2t, SA, SBt)
        C3, S = F.nvidia_transform(C2, 'row', state=SC)
        torch.testing.assert_close(C1, C3.float())