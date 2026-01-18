import itertools
import contextlib
import operator
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_raises, assert_equal
def test_add_128():
    with exc_iter(INT128_VALUES, INT128_VALUES) as it:
        for a, b in it:
            c = a + b
            if not INT128_MIN <= c <= INT128_MAX:
                assert_raises(OverflowError, mt.extint_add_128, a, b)
            else:
                d = mt.extint_add_128(a, b)
                if c != d:
                    assert_equal(d, c)