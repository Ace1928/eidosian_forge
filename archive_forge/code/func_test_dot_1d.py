import numpy
import numpy.linalg as NLA
import pytest
import modin.numpy as np
import modin.numpy.linalg as LA
import modin.pandas as pd
from .utils import assert_scalar_or_array_equal
def test_dot_1d():
    x1 = numpy.random.randint(-100, 100, size=100)
    x2 = numpy.random.randint(-100, 100, size=100)
    numpy_result = numpy.dot(x1, x2)
    x1, x2 = (np.array(x1), np.array(x2))
    modin_result = np.dot(x1, x2)
    assert_scalar_or_array_equal(modin_result, numpy_result)