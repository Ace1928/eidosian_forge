import logging
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..rcparams import rcParams
from .base import dict_to_dataset, infer_stan_dtypes, make_attrs, requires
from .inference_data import InferenceData
def from_cmdstanpy(posterior=None, *, posterior_predictive=None, predictions=None, prior=None, prior_predictive=None, observed_data=None, constant_data=None, predictions_constant_data=None, log_likelihood=None, index_origin=None, coords=None, dims=None, save_warmup=None, dtypes=None):
    """Convert CmdStanPy data into an InferenceData object.

    For a usage example read the
    :ref:`Creating InferenceData section on from_cmdstanpy <creating_InferenceData>`

    Parameters
    ----------
    posterior : CmdStanMCMC object
        CmdStanPy CmdStanMCMC
    posterior_predictive : str, list of str
        Posterior predictive samples for the fit.
    predictions : str, list of str
        Out of sample prediction samples for the fit.
    prior : CmdStanMCMC
        CmdStanPy CmdStanMCMC
    prior_predictive : str, list of str
        Prior predictive samples for the fit.
    observed_data : dict
        Observed data used in the sampling.
    constant_data : dict
        Constant data used in the sampling.
    predictions_constant_data : dict
        Constant data for predictions used in the sampling.
    log_likelihood : str, list of str, dict of {str: str}, optional
        Pointwise log_likelihood for the data. If a dict, its keys should represent var_names
        from the corresponding observed data and its values the stan variable where the
        data is stored. By default, if a variable ``log_lik`` is present in the Stan model,
        it will be retrieved as pointwise log likelihood values. Use ``False``
        or set ``data.log_likelihood`` to false to avoid this behaviour.
    index_origin : int, optional
        Starting value of integer coordinate values. Defaults to the value in rcParam
        ``data.index_origin``.
    coords : dict of str or dict of iterable
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict of str or list of str
        A mapping from variables to a list of coordinate names for the variable.
    save_warmup : bool
        Save warmup iterations into InferenceData object, if found in the input files.
        If not defined, use default defined by the rcParams.
    dtypes: dict or str or cmdstanpy.CmdStanModel
        A dictionary containing dtype information (int, float) for parameters.
        If input is a string, it is assumed to be a model code or path to model code file.
        Model code can extracted from cmdstanpy.CmdStanModel object.

    Returns
    -------
    InferenceData object
    """
    return CmdStanPyConverter(posterior=posterior, posterior_predictive=posterior_predictive, predictions=predictions, prior=prior, prior_predictive=prior_predictive, observed_data=observed_data, constant_data=constant_data, predictions_constant_data=predictions_constant_data, log_likelihood=log_likelihood, index_origin=index_origin, coords=coords, dims=dims, save_warmup=save_warmup, dtypes=dtypes).to_inference_data()