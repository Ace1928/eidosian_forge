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
def test_inplace_floor_division_array_type(self):
    unsupported = {np.dtype(t).type for t in np.typecodes['Complex']}
    for t in self.othertypes:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            x, y, xm = (_.astype(t) for _ in self.uint8data)
            m = xm.mask
            a = arange(10, dtype=t)
            a[-1] = masked
            try:
                x //= a
                xm //= a
                assert_equal(x, y // a)
                assert_equal(xm, y // a)
                assert_equal(xm.mask, mask_or(mask_or(m, a.mask), a == t(0)))
            except TypeError:
                msg = f'Supported type {t} throwing TypeError'
                assert t in unsupported, msg