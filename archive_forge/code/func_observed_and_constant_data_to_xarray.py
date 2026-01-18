import re
from collections import OrderedDict
from copy import deepcopy
from math import ceil
import numpy as np
import xarray as xr
from .. import _log
from ..rcparams import rcParams
from .base import dict_to_dataset, generate_dims_coords, infer_stan_dtypes, make_attrs, requires
from .inference_data import InferenceData
@requires('posterior_model')
@requires(['observed_data', 'constant_data'])
def observed_and_constant_data_to_xarray(self):
    """Convert observed data to xarray."""
    posterior_model = self.posterior_model
    dims = {} if self.dims is None else self.dims
    obs_const_dict = {}
    for group_name in ('observed_data', 'constant_data'):
        names = getattr(self, group_name)
        if names is None:
            continue
        names = [names] if isinstance(names, str) else names
        data = OrderedDict()
        for key in names:
            vals = np.atleast_1d(posterior_model.data[key])
            val_dims = dims.get(key)
            val_dims, coords = generate_dims_coords(vals.shape, key, dims=val_dims, coords=self.coords)
            data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
        obs_const_dict[group_name] = xr.Dataset(data_vars=data, attrs=make_attrs(library=self.stan))
    return obs_const_dict