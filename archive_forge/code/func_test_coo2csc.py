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
def test_coo2csc():
    threshold = 1
    A = torch.randn(128, 128).half().cuda()
    idx = torch.abs(A) >= threshold
    nnz = (idx == 1).sum().item()
    rows, cols = torch.where(idx)
    values = A[idx]
    cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)
    A2 = A * idx
    cscA = F.coo2csc(cooA)
    counts = cscA.colptr[1:] - cscA.colptr[:-1]
    assert counts.numel() == A.shape[1]
    torch.testing.assert_close(counts.long(), (A2 != 0).sum(0))
    idx = A2.t() != 0
    torch.testing.assert_close(A2.t()[idx], cscA.values)