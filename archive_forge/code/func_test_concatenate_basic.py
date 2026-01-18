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
def test_concatenate_basic(self):
    x, y, a10, m1, m2, xm, ym, z, zm, xf = self.d
    assert_equal(np.concatenate((x, y)), concatenate((xm, ym)))
    assert_equal(np.concatenate((x, y)), concatenate((x, y)))
    assert_equal(np.concatenate((x, y)), concatenate((xm, y)))
    assert_equal(np.concatenate((x, y, x)), concatenate((x, ym, x)))