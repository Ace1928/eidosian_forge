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
def infer_dtypes(fit, model=None):
    """Infer dtypes from Stan model code.

    Function strips out generated quantities block and searches for `int`
    dtypes after stripping out comments inside the block.
    """
    if model is None:
        stan_code = fit.get_stancode()
        model_pars = fit.model_pars
    else:
        stan_code = model.program_code
        model_pars = fit.param_names
    dtypes = {key: item for key, item in infer_stan_dtypes(stan_code).items() if key in model_pars}
    return dtypes