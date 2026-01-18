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
def test_cumsumprod_with_output(self):
    xm = array(np.random.uniform(0, 10, 12)).reshape(3, 4)
    xm[:, 0] = xm[0] = xm[-1, -1] = masked
    for funcname in ('cumsum', 'cumprod'):
        npfunc = getattr(np, funcname)
        xmmeth = getattr(xm, funcname)
        output = np.empty((3, 4), dtype=float)
        output.fill(-9999)
        result = npfunc(xm, axis=0, out=output)
        assert_(result is output)
        assert_equal(result, xmmeth(axis=0, out=output))
        output = empty((3, 4), dtype=int)
        result = xmmeth(axis=0, out=output)
        assert_(result is output)