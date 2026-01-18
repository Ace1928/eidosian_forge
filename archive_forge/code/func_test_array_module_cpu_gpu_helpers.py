import numpy
import pytest
from hypothesis import given
from thinc.api import Padded, Ragged, get_width
from thinc.types import ArgsKwargs
from thinc.util import (
from . import strategies
@pytest.mark.parametrize('xp', ALL_XP)
def test_array_module_cpu_gpu_helpers(xp):
    error = "Only numpy and cupy arrays are supported, but found <class 'int'> instead. If get_array_module module wasn't called directly, this might indicate a bug in Thinc."
    with pytest.raises(ValueError, match=error):
        get_array_module(0)
    zeros = xp.zeros((1, 2))
    xp_ = get_array_module(zeros)
    assert xp_ == xp
    if xp == numpy:
        assert is_numpy_array(zeros)
        assert not is_numpy_array((1, 2))
    else:
        assert is_cupy_array(zeros)
        assert not is_cupy_array((1, 2))