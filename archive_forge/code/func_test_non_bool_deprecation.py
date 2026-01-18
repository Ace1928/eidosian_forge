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
def test_non_bool_deprecation(self):
    choices = self.choices
    conditions = self.conditions[:]
    conditions[0] = conditions[0].astype(np.int_)
    assert_raises(TypeError, select, conditions, choices)
    conditions[0] = conditions[0].astype(np.uint8)
    assert_raises(TypeError, select, conditions, choices)
    assert_raises(TypeError, select, conditions, choices)