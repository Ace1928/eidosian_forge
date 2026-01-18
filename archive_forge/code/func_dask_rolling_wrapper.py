from __future__ import annotations
from xarray.core import dtypes, nputils
def dask_rolling_wrapper(moving_func, a, window, min_count=None, axis=-1):
    """Wrapper to apply bottleneck moving window funcs on dask arrays"""
    import dask.array as da
    dtype, fill_value = dtypes.maybe_promote(a.dtype)
    a = a.astype(dtype)
    if axis < 0:
        axis = a.ndim + axis
    depth = {d: 0 for d in range(a.ndim)}
    depth[axis] = (window + 1) // 2
    boundary = {d: fill_value for d in range(a.ndim)}
    ag = da.overlap.overlap(a, depth=depth, boundary=boundary)
    out = da.map_blocks(moving_func, ag, window, min_count=min_count, axis=axis, dtype=a.dtype)
    result = da.overlap.trim_internal(out, depth)
    return result