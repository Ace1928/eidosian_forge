import operator
import numpy as np
from numpy import ndarray, float_
import numpy.core.umath as umath
import numpy.testing
from numpy.testing import (
from .core import mask_or, getmask, masked_array, nomask, masked, filled
from unittest import TestCase
def assert_equal_records(a, b):
    """
    Asserts that two records are equal.

    Pretty crude for now.

    """
    assert_equal(a.dtype, b.dtype)
    for f in a.dtype.names:
        af, bf = (operator.getitem(a, f), operator.getitem(b, f))
        if not af is masked and (not bf is masked):
            assert_equal(operator.getitem(a, f), operator.getitem(b, f))
    return