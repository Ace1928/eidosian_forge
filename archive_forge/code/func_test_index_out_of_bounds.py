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
@pytest.mark.parametrize('idx', [4, -4])
def test_index_out_of_bounds(self, idx):
    with pytest.raises(IndexError, match='out of bounds'):
        np.insert([0, 1, 2], [idx], [3, 4])