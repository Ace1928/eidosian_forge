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
@composite
def draw_lstm_args(draw):
    depth = draw(integers(1, 4))
    dirs = draw(integers(1, 2))
    nO = draw(integers(1, 16)) * dirs
    batch_size = draw(integers(1, 6))
    nI = draw(integers(1, 16))
    return get_lstm_args(depth, dirs, nO, batch_size, nI, draw=draw)