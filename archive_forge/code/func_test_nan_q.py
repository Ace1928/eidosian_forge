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
def test_nan_q(self):
    with pytest.raises(ValueError, match='Percentiles must be in'):
        np.percentile([1, 2, 3, 4.0], np.nan)
    with pytest.raises(ValueError, match='Percentiles must be in'):
        np.percentile([1, 2, 3, 4.0], [np.nan])
    q = np.linspace(1.0, 99.0, 16)
    q[0] = np.nan
    with pytest.raises(ValueError, match='Percentiles must be in'):
        np.percentile([1, 2, 3, 4.0], q)