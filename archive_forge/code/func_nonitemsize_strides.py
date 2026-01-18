import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def nonitemsize_strides(arrs):
    out = []
    for a in arrs:
        a_dtype = a.dtype
        b = np.zeros(a.shape, [('a', a_dtype), ('junk', 'S1')])
        c = b.getfield(a_dtype)
        c[...] = a
        out.append(c)
    return out