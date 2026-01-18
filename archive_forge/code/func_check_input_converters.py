import contextlib
import shutil
import tempfile
from pathlib import Path
import numpy
import pytest
from thinc.api import ArgsKwargs, Linear, Padded, Ragged
from thinc.util import has_cupy, is_cupy_array, is_numpy_array
def check_input_converters(Y, backprop, data, n_args, kwargs_keys, type_):
    assert isinstance(Y, ArgsKwargs)
    assert len(Y.args) == n_args
    assert list(Y.kwargs.keys()) == kwargs_keys
    assert all((isinstance(arg, type_) for arg in Y.args))
    assert all((isinstance(arg, type_) for arg in Y.kwargs.values()))
    dX = backprop(Y)

    def is_supported_backend_array(arr):
        return is_cupy_array(arr) or is_numpy_array(arr)
    input_type = type(data) if not isinstance(data, list) else tuple
    assert isinstance(dX, input_type) or is_supported_backend_array(dX)
    if isinstance(data, dict):
        assert list(dX.keys()) == kwargs_keys
        assert all((is_supported_backend_array(arr) for arr in dX.values()))
    elif isinstance(data, (list, tuple)):
        assert isinstance(dX, tuple)
        assert all((is_supported_backend_array(arr) for arr in dX))
    elif isinstance(data, ArgsKwargs):
        assert len(dX.args) == n_args
        assert list(dX.kwargs.keys()) == kwargs_keys
        assert all((is_supported_backend_array(arg) for arg in dX.args))
        assert all((is_supported_backend_array(arg) for arg in dX.kwargs.values()))
    elif not isinstance(data, numpy.ndarray):
        pytest.fail(f'Bad data type: {dX}')