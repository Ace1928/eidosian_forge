import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
def test_ddof_too_big(self):
    nanfuncs = [np.nanvar, np.nanstd]
    stdfuncs = [np.var, np.std]
    dsize = [len(d) for d in _rdat]
    for nf, rf in zip(nanfuncs, stdfuncs):
        for ddof in range(5):
            with suppress_warnings() as sup:
                sup.record(RuntimeWarning)
                sup.filter(np.ComplexWarning)
                tgt = [ddof >= d for d in dsize]
                res = nf(_ndat, axis=1, ddof=ddof)
                assert_equal(np.isnan(res), tgt)
                if any(tgt):
                    assert_(len(sup.log) == 1)
                else:
                    assert_(len(sup.log) == 0)