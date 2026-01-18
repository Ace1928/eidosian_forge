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
def test_keywords4_ticket_2100(self):

    @vectorize
    def f(**kw):
        res = 1.0
        for _k in kw:
            res *= kw[_k]
        return res
    assert_array_equal(f(a=[1, 2], b=[3, 4]), [3, 8])