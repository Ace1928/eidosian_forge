import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
def test_dtype_from_char(self):
    mat = np.eye(3)
    codes = 'efdgFDG'
    for nf, rf in zip(self.nanfuncs, self.stdfuncs):
        for c in codes:
            with suppress_warnings() as sup:
                if nf in {np.nanstd, np.nanvar} and c in 'FDG':
                    sup.filter(np.ComplexWarning)
                tgt = rf(mat, dtype=c, axis=1).dtype.type
                res = nf(mat, dtype=c, axis=1).dtype.type
                assert_(res is tgt)
                tgt = rf(mat, dtype=c, axis=None).dtype.type
                res = nf(mat, dtype=c, axis=None).dtype.type
                assert_(res is tgt)