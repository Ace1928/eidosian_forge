import logging
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..rcparams import rcParams
from .base import dict_to_dataset, infer_stan_dtypes, make_attrs, requires
from .inference_data import InferenceData
def stats_to_xarray(self, fit):
    """Extract sample_stats from fit."""
    if not (hasattr(fit, 'metadata') or hasattr(fit, 'sampler_vars_cols')):
        return self.sample_stats_to_xarray_pre_v_0_9_68(fit)
    if hasattr(fit, 'metadata') and hasattr(fit.metadata, 'stan_vars_cols') or hasattr(fit, 'stan_vars_cols'):
        return self.sample_stats_to_xarray_pre_v_1_0_0(fit)
    if hasattr(fit, 'metadata') and hasattr(fit.metadata, 'stan_vars_cols'):
        return self.sample_stats_to_xarray_pre_v_1_2_0(fit)
    dtypes = {'divergent__': bool, 'n_leapfrog__': np.int64, 'treedepth__': np.int64, **self.dtypes}
    items = list(fit.method_variables())
    rename_dict = {'divergent': 'diverging', 'n_leapfrog': 'n_steps', 'treedepth': 'tree_depth', 'stepsize': 'step_size', 'accept_stat': 'acceptance_rate'}
    data, data_warmup = _unpack_fit(fit, items, self.save_warmup, self.dtypes)
    for item in items:
        name = re.sub('__$', '', item)
        name = rename_dict.get(name, name)
        data[name] = data.pop(item).astype(dtypes.get(item, float))
        if data_warmup:
            data_warmup[name] = data_warmup.pop(item).astype(dtypes.get(item, float))
    return (dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin), dict_to_dataset(data_warmup, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin))