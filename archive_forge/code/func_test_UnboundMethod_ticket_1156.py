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
def test_UnboundMethod_ticket_1156(self):

    class Foo:
        b = 2

        def bar(self, a):
            return a ** self.b
    assert_array_equal(vectorize(Foo().bar)(np.arange(9)), np.arange(9) ** 2)
    assert_array_equal(vectorize(Foo.bar)(Foo(), np.arange(9)), np.arange(9) ** 2)