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
@pytest.mark.parametrize('dim1', get_test_dims(256, 1024, n=2), ids=id_formatter('dim1'))
@pytest.mark.parametrize('dim2', get_test_dims(256, 1024, n=2), ids=id_formatter('dim2'))
def test_integrated_sparse_decomp(dim1, dim2):
    threshold = 3.0
    formatB = 'col_turing'
    for i in range(k):
        A = torch.randn(dim1, dim2).cuda().half()
        w1 = torch.randn(dim1, dim2).cuda().half()
        out1 = torch.matmul(A, w1.t())
        Cw1, Cw1t, statsw1, statsw1t, coo_tensor = F.double_quant(w1)
        CTw1, Sw1 = F.transform(Cw1, formatB)
        CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)
        C32A, SA = F.transform(CA, 'col32')
        out1_32, Sout1_32 = F.igemmlt(C32A, CTw1, SA, Sw1)
        out2 = F.mm_dequant(out1_32, Sout1_32, statsA, statsw1)
        CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A, threshold=threshold)
        C32A, SA = F.transform(CA, 'col32')
        out1_32, Sout1_32 = F.igemmlt(C32A, CTw1, SA, Sw1)
        out3 = F.mm_dequant(out1_32, Sout1_32, statsA, statsw1)
        assert coo_tensor is not None
        out4 = F.spmm_coo(coo_tensor, w1.t())
        out5 = out3 + out4
        err1 = torch.abs(out1 - out2).mean().item()
        err2 = torch.abs(out1 - out5).mean().item()
        assert err2 < err1