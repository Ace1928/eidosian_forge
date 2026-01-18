import warnings
from typing import Optional
from ..rcparams import rcParams
from .base import dict_to_dataset, requires
from .inference_data import WARMUP_TAG, InferenceData
from_pytree = from_dict
def data_to_xarray(self, data, group, dims=None):
    """Convert data to xarray."""
    if not isinstance(data, dict):
        raise TypeError(f'DictConverter.{group} is not a dictionary')
    if dims is None:
        dims = {} if self.dims is None else self.dims
    return dict_to_dataset(data, library=None, coords=self.coords, dims=self.dims, default_dims=[], attrs=self.attrs, index_origin=self.index_origin)