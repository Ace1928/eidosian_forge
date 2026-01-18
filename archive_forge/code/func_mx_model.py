from typing import cast
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_mxnet
from thinc.types import Array1d, Array2d, IntsXd
from thinc.util import to_categorical
from ..util import check_input_converters, make_tempdir
@pytest.fixture
def mx_model(n_hidden: int, input_size: int, X: Array2d):
    import mxnet as mx
    mx_model = mx.gluon.nn.Sequential()
    mx_model.add(mx.gluon.nn.Dense(n_hidden), mx.gluon.nn.LayerNorm(), mx.gluon.nn.Dense(n_hidden, activation='relu'), mx.gluon.nn.LayerNorm(), mx.gluon.nn.Dense(10, activation='softrelu'))
    mx_model.initialize()
    return mx_model