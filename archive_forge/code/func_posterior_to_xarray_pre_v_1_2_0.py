import logging
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..rcparams import rcParams
from .base import dict_to_dataset, infer_stan_dtypes, make_attrs, requires
from .inference_data import InferenceData
def posterior_to_xarray_pre_v_1_2_0(self):
    items = list(self.posterior.metadata.stan_vars_cols)
    if self.posterior_predictive is not None:
        try:
            items = _filter(items, self.posterior_predictive)
        except ValueError:
            pass
    if self.predictions is not None:
        try:
            items = _filter(items, self.predictions)
        except ValueError:
            pass
    if self.log_likelihood is not None:
        try:
            items = _filter(items, self.log_likelihood)
        except ValueError:
            pass
    valid_cols = []
    for item in items:
        if hasattr(self.posterior, 'metadata'):
            if item in self.posterior.metadata.stan_vars_cols:
                valid_cols.append(item)
    data, data_warmup = _unpack_fit_pre_v_1_2_0(self.posterior, items, self.save_warmup, self.dtypes)
    dims = deepcopy(self.dims) if self.dims is not None else {}
    coords = deepcopy(self.coords) if self.coords is not None else {}
    return (dict_to_dataset(data, library=self.cmdstanpy, coords=coords, dims=dims, index_origin=self.index_origin), dict_to_dataset(data_warmup, library=self.cmdstanpy, coords=coords, dims=dims, index_origin=self.index_origin))