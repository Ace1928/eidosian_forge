from io import BytesIO
import numpy as np
from numpy.testing import assert_array_equal
from .. import Nifti1Header, Nifti1Image
from ..arraywriters import ScalingError
from ..casting import best_float, sctypes, type_info, ulp
from ..spatialimages import HeaderDataError, supported_np_types
def big_bad_ulp(arr):
    """Return array of ulp values for values in `arr`

    I haven't thought about whether the vectorized log2 here could lead to
    incorrect rounding; this only needs to be ballpark

    This function might be used in nipy/io/tests/test_image_io.py

    Parameters
    ----------
    arr : array
        floating point array

    Returns
    -------
    ulps : array
        ulp values for each element of arr
    """
    arr = np.asarray(arr)
    info = type_info(arr.dtype)
    working_arr = np.abs(arr.astype(BFT))
    fl2 = np.zeros_like(working_arr) + info['minexp']
    nzs = working_arr > 0
    fl2[nzs] = np.floor(np.log(working_arr[nzs]) / LOGe2)
    fl2 = np.clip(fl2, info['minexp'], np.inf)
    return 2 ** (fl2 - info['nmant'])