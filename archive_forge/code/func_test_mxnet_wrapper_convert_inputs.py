from typing import cast
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_mxnet
from thinc.types import Array1d, Array2d, IntsXd
from thinc.util import to_categorical
from ..util import check_input_converters, make_tempdir
@pytest.mark.skipif(not has_mxnet, reason='needs MXNet')
@pytest.mark.parametrize('data,n_args,kwargs_keys', [(numpy.zeros((2, 3), dtype='f'), 1, []), ([numpy.zeros((2, 3), dtype='f'), numpy.zeros((2, 3), dtype='f')], 2, []), ((numpy.zeros((2, 3), dtype='f'), numpy.zeros((2, 3), dtype='f')), 2, []), ({'a': numpy.zeros((2, 3), dtype='f'), 'b': numpy.zeros((2, 3), dtype='f')}, 0, ['a', 'b']), (ArgsKwargs((numpy.zeros((2, 3), dtype='f'), numpy.zeros((2, 3), dtype='f')), {'c': numpy.zeros((2, 3), dtype='f')}), 2, ['c'])])
def test_mxnet_wrapper_convert_inputs(data, n_args, kwargs_keys):
    import mxnet as mx
    mx_model = mx.gluon.nn.Sequential()
    mx_model.add(mx.gluon.nn.Dense(12))
    mx_model.initialize()
    model = MXNetWrapper(mx_model)
    convert_inputs = model.attrs['convert_inputs']
    Y, backprop = convert_inputs(model, data, is_train=True)
    check_input_converters(Y, backprop, data, n_args, kwargs_keys, mx.nd.NDArray)