import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def check_var_expect(distfn, arg, m, v, msg):
    dist_looser_tolerances = {'rv_histogram_instance', 'ksone'}
    kwargs = {'rtol': 5e-06} if msg in dist_looser_tolerances else {}
    if np.isfinite(v):
        m2 = distfn.expect(lambda x: x * x, arg)
        npt.assert_allclose(m2, v + m * m, **kwargs)