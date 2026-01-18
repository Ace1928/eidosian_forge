from typing import cast
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_mxnet
from thinc.types import Array1d, Array2d, IntsXd
from thinc.util import to_categorical
from ..util import check_input_converters, make_tempdir
@pytest.mark.skipif(not has_mxnet, reason='needs MXNet')
def test_mxnet_wrapper_roundtrip_conversion():
    import mxnet as mx
    xp_tensor = numpy.zeros((2, 3), dtype='f')
    mx_tensor = xp2mxnet(xp_tensor)
    assert isinstance(mx_tensor, mx.nd.NDArray)
    new_xp_tensor = mxnet2xp(mx_tensor)
    assert numpy.array_equal(xp_tensor, new_xp_tensor)