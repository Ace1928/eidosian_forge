import torch
import functools
from torch.testing import make_tensor
from functorch.experimental.control_flow import map
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
def simple_map(xs, y0, y1):

    def f(x, y0, y1):
        return inner_f(x, y0, y1)
    return map(f, xs, y0, y1)