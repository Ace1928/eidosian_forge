import math
import einops
import pytest
import torch
from torch import nn
import bitsandbytes as bnb
from tests.helpers import id_formatter
def assert_all_approx_close(a, b, atol=1e-08, rtol=1e-05, count=10):
    idx = torch.isclose(a, b, rtol=rtol, atol=atol)
    sumval = (idx == 0).sum().item()
    if sumval > count:
        print(f'Too many values not close: assert {sumval} < {count}')
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)