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
@pytest.mark.skip('Row scale has some bugs for ampere')
def test_managed():
    n = 32 * 10
    A = F.get_paged(n, n, dtype=torch.float32)
    B = F.get_paged(n, n, dtype=torch.uint8)
    B2 = F.get_paged(n, n, dtype=torch.float32)
    assert A.is_paged
    assert B.is_paged
    assert A.page_deviceid == 0
    assert B.page_deviceid == 0
    F.fill(A, 17.0)
    F.fill(B, 17)
    F.fill(B2, 2)
    assert (A == 17).sum().item() == n * n
    assert (B == 17).sum().item() == n * n
    C = A * B.float()
    assert (C == 289).sum().item() == n * n
    F._mul(A, B2)
    F._mul(A, B2)
    F._mul(A, B2)
    assert (A == 17 * 2 ** 3).sum().item() == n * n