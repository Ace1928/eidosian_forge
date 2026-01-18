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
def test_failed_itemsetting(self):
    with pytest.raises(TypeError):
        np.fromiter([1, None, 3], dtype=int)
    iterable = ((2, 3, 4) for i in range(5))
    with pytest.raises(ValueError):
        np.fromiter(iterable, dtype=np.dtype((int, 2)))