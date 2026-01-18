import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
def test_anom(self):
    a = masked_array(np.arange(1, 7).reshape(2, 3))
    assert_almost_equal(a.anom(), [[-2.5, -1.5, -0.5], [0.5, 1.5, 2.5]])
    assert_almost_equal(a.anom(axis=0), [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
    assert_almost_equal(a.anom(axis=1), [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    a.mask = [[0, 0, 1], [0, 1, 0]]
    mval = -99
    assert_almost_equal(a.anom().filled(mval), [[-2.25, -1.25, mval], [0.75, mval, 2.75]])
    assert_almost_equal(a.anom(axis=0).filled(mval), [[-1.5, 0.0, mval], [1.5, mval, 0.0]])
    assert_almost_equal(a.anom(axis=1).filled(mval), [[-0.5, 0.5, mval], [-1.0, mval, 1.0]])