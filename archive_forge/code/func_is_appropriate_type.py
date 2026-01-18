imported when doing ``import xarray_einstats``.
import warnings
from collections.abc import Hashable
import einops
import xarray as xr
def is_appropriate_type(self, tensor):
    """Recognizes tensors it can handle."""
    return isinstance(tensor, self.dsar.core.Array)