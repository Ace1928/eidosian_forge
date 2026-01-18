import numba
import numpy as np
import xarray as xr
from . import _remove_indexes_to_reduce, sort
@numba.guvectorize(['void(uint8[:], uint8[:], uint8[:])', 'void(uint16[:], uint16[:], uint16[:])', 'void(uint32[:], uint32[:], uint32[:])', 'void(uint64[:], uint64[:], uint64[:])', 'void(int8[:], int8[:], int8[:])', 'void(int16[:], int16[:], int16[:])', 'void(int32[:], int32[:], int32[:])', 'void(int64[:], int64[:], int64[:])', 'void(float32[:], float32[:], float32[:])', 'void(float64[:], float64[:], float64[:])'], '(n),(m)->(m)', cache=True, target='parallel', nopython=True)
def hist_ufunc(data, bin_edges, res):
    """Use :func:`numba.guvectorize` to convert numpy histogram into a ufunc.

    Notes
    -----
    ``bin_edges`` is a required argument because it is needed to have a valid call
    signature. The shape of the output must be generated from the dimensions available
    in the inputs; they can be in different order, duplicated or reduced, but the output
    can't introduce new dimensions.
    """
    m = len(bin_edges)
    res[:] = 0
    aux, _ = np.histogram(data, bins=bin_edges)
    for i in numba.prange(m - 1):
        res[i] = aux[i]