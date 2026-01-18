import torch
import functools
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
from torch.testing._internal.autograd_function_db import (
from torch import Tensor
from torch.types import Number
from typing import *  # noqa: F403
import torch._custom_ops as custom_ops
def sample_inputs_numpy_nms(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    N = 64
    xs = make_arg([N], low=0, high=28)
    dx = make_arg([N], low=0, high=4)
    ys = make_arg([N], low=0, high=28)
    dy = make_arg([N], low=0, high=4)
    boxes = torch.stack([xs, ys, xs + dx, ys + dy], dim=1).requires_grad_(requires_grad)
    scores = make_arg([N], low=0, high=1, requires_grad=requires_grad)
    iou_threshold = make_arg([], low=0, high=1).item()
    yield SampleInput(boxes, args=(scores, iou_threshold))