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
def test_fillvalue_implicit_structured_array(self):
    ndtype = ('b', float)
    adtype = ('a', float)
    a = array([(1.0,), (2.0,)], mask=[(False,), (False,)], fill_value=(np.nan,), dtype=np.dtype([adtype]))
    b = empty(a.shape, dtype=[adtype, ndtype])
    b['a'] = a['a']
    b['a'].set_fill_value(a['a'].fill_value)
    f = b._fill_value[()]
    assert_(np.isnan(f[0]))
    assert_equal(f[-1], default_fill_value(1.0))