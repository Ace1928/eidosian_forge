import logging
from typing import Callable, Optional
import warnings
import numpy as np
from packaging import version
from .. import utils
from ..rcparams import rcParams
from .base import dict_to_dataset, requires
from .inference_data import InferenceData
def priors_to_xarray(self):
    """Convert prior samples (and if possible prior predictive too) to xarray."""
    if self.prior is None:
        return {'prior': None, 'prior_predictive': None}
    if self.posterior is not None:
        prior_vars = list(self.posterior.get_samples().keys())
        prior_predictive_vars = [key for key in self.prior.keys() if key not in prior_vars]
    else:
        prior_vars = self.prior.keys()
        prior_predictive_vars = None
    priors_dict = {group: None if var_names is None else dict_to_dataset({k: utils.expand_dims(np.squeeze(self.prior[k].detach().cpu().numpy())) for k in var_names}, library=self.pyro, coords=self.coords, dims=self.dims) for group, var_names in zip(('prior', 'prior_predictive'), (prior_vars, prior_predictive_vars))}
    return priors_dict