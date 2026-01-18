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
def test_negative_resize(self):
    A = np.arange(0, 10, dtype=np.float32)
    new_shape = (-10, -1)
    with pytest.raises(ValueError, match='negative'):
        np.resize(A, new_shape=new_shape)