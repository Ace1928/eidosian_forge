imported when doing ``import xarray_einstats``.
import warnings
from collections.abc import Hashable
import einops
import xarray as xr
def raw_reduce(*args, **kwargs):
    """Wrap einops.reduce.

    DEPRECATED
    """
    warnings.warn('raw_reduce has been deprecated. Its functionality has been merged into reduce', DeprecationWarning)
    return reduce(*args, **kwargs)