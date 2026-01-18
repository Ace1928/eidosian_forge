import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
@pytest.mark.parametrize('string', ['S', 'U'])
@pytest.mark.parametrize('floating', ['e', 'f', 'd', 'g'])
def test_float_to_string(self, floating, string):
    assert np.can_cast(floating, string)
    assert np.can_cast(floating, f'{string}100')