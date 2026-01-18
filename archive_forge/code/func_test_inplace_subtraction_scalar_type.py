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
def test_inplace_subtraction_scalar_type(self):
    for t in self.othertypes:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            x, y, xm = (_.astype(t) for _ in self.uint8data)
            x -= t(1)
            assert_equal(x, y - t(1))
            xm -= t(1)
            assert_equal(xm, y - t(1))