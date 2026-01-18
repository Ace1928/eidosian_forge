import logging
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..rcparams import rcParams
from .base import dict_to_dataset, infer_stan_dtypes, make_attrs, requires
from .inference_data import InferenceData
def predictive_to_xarray(self, names, fit):
    """Convert predictive samples to xarray."""
    predictive = _as_set(names)
    if not (hasattr(fit, 'metadata') or hasattr(fit, 'stan_vars_cols')):
        valid_cols = _filter_columns(fit.column_names, predictive)
        data, data_warmup = _unpack_frame(fit, fit.column_names, valid_cols, self.save_warmup, self.dtypes)
    elif hasattr(fit, 'metadata') and hasattr(fit.metadata, 'sample_vars_cols') or hasattr(fit, 'stan_vars_cols'):
        data, data_warmup = _unpack_fit_pre_v_1_0_0(fit, predictive, self.save_warmup, self.dtypes)
    elif hasattr(fit, 'metadata') and hasattr(fit.metadata, 'stan_vars_cols'):
        data, data_warmup = _unpack_fit_pre_v_1_2_0(fit, predictive, self.save_warmup, self.dtypes)
    else:
        data, data_warmup = _unpack_fit(fit, predictive, self.save_warmup, self.dtypes)
    return (dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin), dict_to_dataset(data_warmup, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin))