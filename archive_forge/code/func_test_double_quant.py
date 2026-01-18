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
@pytest.mark.parametrize('dim1', get_test_dims(1, 4 * 1024, n=2), ids=id_formatter('dim1'))
@pytest.mark.parametrize('dim2', get_test_dims(1, 4 * 1024, n=2), ids=id_formatter('dim2'))
def test_double_quant(dim1, dim2):
    for i in range(k):
        A = torch.randn(dim1, dim2, device='cuda').half()
        out_col1, Scol = F.vectorwise_quant(A, dim=0)
        out_row1, Srow = F.vectorwise_quant(A, dim=1)
        CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)
        torch.testing.assert_close(CA, out_row1, atol=1, rtol=0)
        torch.testing.assert_close(CAt, out_col1, atol=1, rtol=0)
        n = CAt.numel()
        num_not_close_rows = (torch.isclose(CA, out_row1, atol=1) == 0).sum().item()
        num_not_close_cols = (torch.isclose(CAt, out_col1, atol=1) == 0).sum().item()
        min_error = 1 / 500
        if num_not_close_cols > min_error * n:
            print(f'Min error exceeded {num_not_close_cols} elements are different. Error: {num_not_close_cols / n:.4f}')
            assert False
        if num_not_close_rows > min_error * n:
            print(f'Min error exceeded {num_not_close_rows} elements are different. Error: {num_not_close_rows / n:.4f}')
            assert False
        torch.testing.assert_close(Srow.flatten().float(), statsA)
        torch.testing.assert_close(Scol.flatten().float(), statsAt)