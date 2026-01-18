import logging
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..rcparams import rcParams
from .base import dict_to_dataset, infer_stan_dtypes, make_attrs, requires
from .inference_data import InferenceData
def sample_stats_to_xarray_pre_v_1_2_0(self, fit):
    dtypes = {'divergent__': bool, 'n_leapfrog__': np.int64, 'treedepth__': np.int64, **self.dtypes}
    items = list(fit.metadata.method_vars_cols.keys())
    rename_dict = {'divergent': 'diverging', 'n_leapfrog': 'n_steps', 'treedepth': 'tree_depth', 'stepsize': 'step_size', 'accept_stat': 'acceptance_rate'}
    data, data_warmup = _unpack_fit_pre_v_1_2_0(fit, items, self.save_warmup, self.dtypes)
    for item in items:
        name = re.sub('__$', '', item)
        name = rename_dict.get(name, name)
        data[name] = data.pop(item).astype(dtypes.get(item, float))
        if data_warmup:
            data_warmup[name] = data_warmup.pop(item).astype(dtypes.get(item, float))
    return (dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin), dict_to_dataset(data_warmup, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin))