import functools
import torch
from torch._inductor.compile_fx import fake_tensor_prop
from ..._dynamo.utils import counters
from .. import config
from ..pattern_matcher import (
def addmm_fuse_replacement_second(inp, w1, w2, w3, b1, b2, b3):
    cat_w = torch.cat((w1, w2, w3), dim=1)
    cat_b = torch.cat((b1, b2, b3))
    return aten.addmm(cat_b, inp, cat_w).chunk(3, dim=1)