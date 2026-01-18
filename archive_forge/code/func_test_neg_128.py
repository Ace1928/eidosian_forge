import itertools
import contextlib
import operator
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_raises, assert_equal
def test_neg_128():
    with exc_iter(INT128_VALUES) as it:
        for a, in it:
            b = -a
            c = mt.extint_neg_128(a)
            if b != c:
                assert_equal(c, b)