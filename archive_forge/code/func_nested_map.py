import torch
import functools
from torch.testing import make_tensor
from functorch.experimental.control_flow import map
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
def nested_map(xs, y0, y1):

    def f1(xx, y0, y1):

        def f2(x, y0, y1):
            return inner_f(x, y0, y1)
        return map(f2, xx, y0, y1)
    return map(f1, xs, y0, y1)