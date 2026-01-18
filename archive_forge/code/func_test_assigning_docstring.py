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
def test_assigning_docstring(self):

    def foo(x):
        """Original documentation"""
        return x
    f = vectorize(foo)
    assert_equal(f.__doc__, foo.__doc__)
    doc = 'Provided documentation'
    f = vectorize(foo, doc=doc)
    assert_equal(f.__doc__, doc)