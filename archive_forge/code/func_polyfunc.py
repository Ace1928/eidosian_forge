import numpy as np
from numpy.testing import assert_, assert_allclose
import pytest
from scipy.special import _ufuncs
import scipy.special._orthogonal as orth
from scipy.special._testutils import FuncData
def polyfunc(*p):
    p = (p[0].astype(int),) + p[1:]
    kw = dict(sig='l' + (len(p) - 1) * 'd' + '->d')
    return func(*p, **kw)