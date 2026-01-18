import numpy
import pytest
from thinc.api import (
from thinc.backends import context_pools
from thinc.compat import has_cupy_gpu, has_torch, has_torch_amp, has_torch_mps_gpu
from thinc.layers.pytorchwrapper import PyTorchWrapper_v3
from thinc.shims.pytorch import (
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.util import get_torch_default_device
from ..util import check_input_converters, make_tempdir
def check_learns_zero_output(model, sgd, X, Y):
    """Check we can learn to output a zero vector"""
    Yh, get_dX = model.begin_update(X)
    dYh = (Yh - Y) / Yh.shape[0]
    dX = get_dX(dYh)
    model.finish_update(sgd)
    prev = numpy.abs(Yh.sum())
    for i in range(100):
        Yh, get_dX = model.begin_update(X)
        total = numpy.abs(Yh.sum())
        dX = get_dX(Yh - Y)
        model.finish_update(sgd)
    assert total < prev