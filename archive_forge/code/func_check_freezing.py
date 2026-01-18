import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def check_freezing(distfn, args):
    if isinstance(distfn, stats.rv_continuous):
        locscale = {'loc': 1, 'scale': 2}
    else:
        locscale = {'loc': 1}
    rv = distfn(*args, **locscale)
    assert rv.a == distfn(*args).a
    assert rv.b == distfn(*args).b