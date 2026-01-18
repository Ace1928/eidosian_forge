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
@pytest.mark.parametrize('dtype', [int, object])
@pytest.mark.parametrize(['count', 'error_index'], [(10, 5), (10, 9)])
def test_2592(self, count, error_index, dtype):
    iterable = self.load_data(count, error_index)
    with pytest.raises(NIterError):
        np.fromiter(iterable, dtype=dtype, count=count)