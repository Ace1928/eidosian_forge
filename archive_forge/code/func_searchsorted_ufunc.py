import numba
import numpy as np
import xarray as xr
from . import _remove_indexes_to_reduce, sort
@numba.guvectorize(['void(uint8[:], uint8[:], uint8[:])', 'void(uint16[:], uint16[:], uint16[:])', 'void(uint32[:], uint32[:], uint32[:])', 'void(uint64[:], uint64[:], uint64[:])', 'void(int8[:], int8[:], int8[:])', 'void(int16[:], int16[:], int16[:])', 'void(int32[:], int32[:], int32[:])', 'void(int64[:], int64[:], int64[:])', 'void(float32[:], float32[:], float32[:])', 'void(float64[:], float64[:], float64[:])'], '(n),(m)->(m)', cache=True, target='parallel', nopython=True)
def searchsorted_ufunc(da, v, res):
    """Use :func:`numba.guvectorize` to convert numpy searchsorted into a vectorized ufunc.

    Notes
    -----
    As of now, its only intended use is in for `ecdf`, so the `side` is
    hardcoded and the rest of the library will assume so.
    """
    res[:] = np.searchsorted(da, v, side='right')