import functools
import itertools
import math
import numpy
from numpy.testing import (assert_equal, assert_allclose,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from scipy.ndimage._filters import _gaussian_kernel1d
from . import types, float_types, complex_types
@pytest.mark.parametrize('dtype_kernel', types)
@pytest.mark.parametrize('dtype_input', complex_types)
@pytest.mark.parametrize('dtype_output', complex_types)
def test_correlate_complex_input(self, dtype_input, dtype_kernel, dtype_output):
    kernel = numpy.array([[1, 0], [0, 1]], dtype_kernel)
    array = numpy.array([[1, 2j, 3], [1 + 4j, 5, 6j]], dtype_input)
    self._validate_complex(array, kernel, dtype_output)