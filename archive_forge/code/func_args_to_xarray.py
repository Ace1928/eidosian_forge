import warnings
from collections import OrderedDict
import numpy as np
import xarray as xr
from .. import utils
from .base import dict_to_dataset, generate_dims_coords, make_attrs
from .inference_data import InferenceData
def args_to_xarray(self):
    """Convert emcee args to observed and constant_data xarray Datasets."""
    dims = {} if self.dims is None else self.dims
    if self.arg_groups is None:
        self.arg_groups = ['observed_data' for _ in self.arg_names]
    if len(self.arg_names) != len(self.arg_groups):
        raise ValueError('arg_names and arg_groups must have the same length, or arg_groups be None')
    arg_groups_set = set(self.arg_groups)
    bad_groups = [group for group in arg_groups_set if group not in ('observed_data', 'constant_data')]
    if bad_groups:
        raise SyntaxError(f"all arg_groups values should be either 'observed_data' or 'constant_data' , not {bad_groups}")
    obs_const_dict = {group: OrderedDict() for group in arg_groups_set}
    for idx, (arg_name, group) in enumerate(zip(self.arg_names, self.arg_groups)):
        arg_array = np.atleast_1d(self.sampler.log_prob_fn.args[idx] if hasattr(self.sampler, 'log_prob_fn') else self.sampler.args[idx])
        arg_dims = dims.get(arg_name)
        arg_dims, coords = generate_dims_coords(arg_array.shape, arg_name, dims=arg_dims, coords=self.coords, index_origin=self.index_origin)
        coords = {key: xr.IndexVariable((key,), data=coords[key]) for key in arg_dims}
        obs_const_dict[group][arg_name] = xr.DataArray(arg_array, dims=arg_dims, coords=coords)
    for key, values in obs_const_dict.items():
        obs_const_dict[key] = xr.Dataset(data_vars=values, attrs=make_attrs(library=self.emcee))
    return obs_const_dict