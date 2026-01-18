import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
def check_invpascal(n, kind, exact):
    ip = invpascal(n, kind=kind, exact=exact)
    p = pascal(n, kind=kind, exact=exact)
    e = ip.astype(object).dot(p.astype(object))
    assert_array_equal(e, eye(n), err_msg='n=%d  kind=%r exact=%r' % (n, kind, exact))