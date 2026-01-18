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
def test_boolean_edgecase(self):
    a = np.array([], dtype='int32')
    b = np.array([], dtype='bool')
    c = a[b]
    assert_equal(c, [])
    assert_equal(c.dtype, np.dtype('int32'))