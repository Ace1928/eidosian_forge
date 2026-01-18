import inspect
import platform
from typing import Tuple, cast
import numpy
import pytest
from hypothesis import given, settings
from hypothesis.strategies import composite, integers
from numpy.testing import assert_allclose
from packaging.version import Version
from thinc.api import (
from thinc.backends._custom_kernels import KERNELS, KERNELS_LIST, compile_mmh
from thinc.compat import has_cupy_gpu, has_torch, torch_version
from thinc.types import Floats2d
from thinc.util import torch2xp, xp2torch
from .. import strategies
from ..strategies import arrays_BI, ndarrays_of_shape
def create_pytorch_funcs():
    import math
    import torch

    def torch_relu(x):
        return torch.nn.functional.relu(x)

    def torch_relu_k(x):
        return torch.nn.functional.relu6(x)

    def torch_hard_sigmoid(x):
        return torch.clip(x * 0.2 + 0.5, 0, 1)

    def torch_hard_tanh(x):
        return torch.nn.functional.hardtanh(x)

    def torch_mish(x):
        return torch.nn.functional.mish(x)

    def torch_swish(x):
        return torch.nn.functional.silu(x)

    def torch_hard_swish(x):
        return x * torch_hard_sigmoid(x)

    def torch_hard_swish_mobilenet(x):
        return torch.nn.functional.hardswish(x)

    def torch_sigmoid(x):
        return torch.sigmoid(x)

    def torch_dish(x):
        return 0.5 * x * (x / (1 + x * x).sqrt() + 1)

    def torch_gelu_approx(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    def torch_gelu(x):
        return torch.nn.functional.gelu(x)
    return [('relu', torch_relu), ('relu_k', torch_relu_k), ('hard_sigmoid', torch_hard_sigmoid), ('hard_tanh', torch_hard_tanh), ('mish', torch_mish), ('swish', torch_swish), ('hard_swish', torch_hard_swish), ('hard_swish_mobilenet', torch_hard_swish_mobilenet), ('dish', torch_dish), ('gelu_approx', torch_gelu_approx), ('gelu', torch_gelu), ('sigmoid', torch_sigmoid)]