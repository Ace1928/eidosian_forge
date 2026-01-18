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
def test_extract_outliers():
    for i in range(k):
        shapeA = (4096, 4096 * 4)
        idx = torch.unique(torch.randint(0, shapeA[1], size=(10,)).int()).cuda()
        A = torch.randint(-128, 127, size=shapeA, device='cuda').to(torch.int8)
        outliers1 = A[:, idx.long()]
        CA, SA = F.transform(A, 'col_turing')
        outliers2 = F.extract_outliers(CA, SA, idx)
        assert outliers2.shape[0] == shapeA[0]
        assert outliers2.shape[1] == idx.numel()
        torch.testing.assert_close(outliers1, outliers2)
        CA, SA = F.transform(A, 'col_ampere')
        outliers2 = F.extract_outliers(CA, SA, idx)
        assert outliers2.shape[0] == shapeA[0]
        assert outliers2.shape[1] == idx.numel()
        torch.testing.assert_close(outliers1, outliers2)