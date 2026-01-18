import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def check_named_args(distfn, x, shape_args, defaults, meths):
    signature = _getfullargspec(distfn._parse_args)
    npt.assert_(signature.varargs is None)
    npt.assert_(signature.varkw is None)
    npt.assert_(not signature.kwonlyargs)
    npt.assert_(list(signature.defaults) == list(defaults))
    shape_argnames = signature.args[:-len(defaults)]
    if distfn.shapes:
        shapes_ = distfn.shapes.replace(',', ' ').split()
    else:
        shapes_ = ''
    npt.assert_(len(shapes_) == distfn.numargs)
    npt.assert_(len(shapes_) == len(shape_argnames))
    shape_args = list(shape_args)
    vals = [meth(x, *shape_args) for meth in meths]
    npt.assert_(np.all(np.isfinite(vals)))
    names, a, k = (shape_argnames[:], shape_args[:], {})
    while names:
        k.update({names.pop(): a.pop()})
        v = [meth(x, *a, **k) for meth in meths]
        npt.assert_array_equal(vals, v)
        if 'n' not in k.keys():
            npt.assert_equal(distfn.moment(1, *a, **k), distfn.moment(1, *shape_args))
    k.update({'kaboom': 42})
    assert_raises(TypeError, distfn.cdf, x, **k)