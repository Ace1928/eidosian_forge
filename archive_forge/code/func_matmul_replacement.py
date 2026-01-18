import functools
import torch
from torch._inductor.compile_fx import fake_tensor_prop
from ..._dynamo.utils import counters
from .. import config
from ..pattern_matcher import (
def matmul_replacement(inp, w1, w2, w3):
    cat_t = torch.cat((w1, w2, w3), dim=1)
    mm = inp @ cat_t
    return mm.chunk(3, dim=1)