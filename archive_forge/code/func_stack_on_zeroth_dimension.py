imported when doing ``import xarray_einstats``.
import warnings
from collections.abc import Hashable
import einops
import xarray as xr
def stack_on_zeroth_dimension(self, tensors: list):
    return self.dsar.stack(tensors)