import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
@pytest.mark.parametrize('dtype', ['V0', 'V3', 'V10'])
def test_compare_unstructured_voids(self, dtype):
    zeros = np.zeros(3, dtype=dtype)
    assert_array_equal(zeros, zeros)
    assert not (zeros != zeros).any()
    if dtype == 'V0':
        return
    nonzeros = np.array([b'1', b'2', b'3'], dtype=dtype)
    assert not (zeros == nonzeros).any()
    assert (zeros != nonzeros).all()