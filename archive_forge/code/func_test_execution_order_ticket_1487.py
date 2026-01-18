import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
def test_execution_order_ticket_1487(self):
    f1 = vectorize(lambda x: x)
    res1a = f1(np.arange(3))
    res1b = f1(np.arange(0.1, 3))
    f2 = vectorize(lambda x: x)
    res2b = f2(np.arange(0.1, 3))
    res2a = f2(np.arange(3))
    assert_equal(res1a, res2a)
    assert_equal(res1b, res2b)