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
def test_fweights(self):
    assert_allclose(cov(self.x2, fweights=self.frequencies), cov(self.x2_repeats))
    assert_allclose(cov(self.x1, fweights=self.frequencies), self.res2)
    assert_allclose(cov(self.x1, fweights=self.unit_frequencies), self.res1)
    nonint = self.frequencies + 0.5
    assert_raises(TypeError, cov, self.x1, fweights=nonint)
    f = np.ones((2, 3), dtype=np.int_)
    assert_raises(RuntimeError, cov, self.x1, fweights=f)
    f = np.ones(2, dtype=np.int_)
    assert_raises(RuntimeError, cov, self.x1, fweights=f)
    f = -1 * np.ones(3, dtype=np.int_)
    assert_raises(ValueError, cov, self.x1, fweights=f)