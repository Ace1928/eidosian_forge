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
def test_minmax_funcs_with_output(self):
    mask = np.random.rand(12).round()
    xm = array(np.random.uniform(0, 10, 12), mask=mask)
    xm.shape = (3, 4)
    for funcname in ('min', 'max'):
        npfunc = getattr(np, funcname)
        mafunc = getattr(numpy.ma.core, funcname)
        nout = np.empty((4,), dtype=int)
        try:
            result = npfunc(xm, axis=0, out=nout)
        except MaskError:
            pass
        nout = np.empty((4,), dtype=float)
        result = npfunc(xm, axis=0, out=nout)
        assert_(result is nout)
        nout.fill(-999)
        result = mafunc(xm, axis=0, out=nout)
        assert_(result is nout)