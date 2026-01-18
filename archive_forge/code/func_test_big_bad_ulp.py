from io import BytesIO
import numpy as np
from numpy.testing import assert_array_equal
from .. import Nifti1Header, Nifti1Image
from ..arraywriters import ScalingError
from ..casting import best_float, sctypes, type_info, ulp
from ..spatialimages import HeaderDataError, supported_np_types
def test_big_bad_ulp():
    for ftype in (np.float32, np.float64):
        ti = type_info(ftype)
        fi = np.finfo(ftype)
        min_ulp = 2 ** (ti['minexp'] - ti['nmant'])
        in_arr = np.zeros((10,), dtype=ftype)
        in_arr = np.array([0, 0, 1, 2, 4, 5, -5, -np.inf, np.inf], dtype=ftype)
        out_arr = [min_ulp, min_ulp, fi.eps, fi.eps * 2, fi.eps * 4, fi.eps * 4, fi.eps * 4, np.inf, np.inf]
        assert_array_equal(big_bad_ulp(in_arr).astype(ftype), out_arr)