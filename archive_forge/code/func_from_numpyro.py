import logging
from typing import Callable, Optional
import numpy as np
from .. import utils
from ..rcparams import rcParams
from .base import dict_to_dataset, requires
from .inference_data import InferenceData
def from_numpyro(posterior=None, *, prior=None, posterior_predictive=None, predictions=None, constant_data=None, predictions_constant_data=None, log_likelihood=None, index_origin=None, coords=None, dims=None, pred_dims=None, num_chains=1):
    """Convert NumPyro data into an InferenceData object.

    For a usage example read the
    :ref:`Creating InferenceData section on from_numpyro <creating_InferenceData>`

    Parameters
    ----------
    posterior : numpyro.mcmc.MCMC
        Fitted MCMC object from NumPyro
    prior: dict
        Prior samples from a NumPyro model
    posterior_predictive : dict
        Posterior predictive samples for the posterior
    predictions: dict
        Out of sample predictions
    constant_data: dict
        Dictionary containing constant data variables mapped to their values.
    predictions_constant_data: dict
        Constant data used for out-of-sample predictions.
    index_origin : int, optional
    coords : dict[str] -> list[str]
        Map of dimensions to coordinates
    dims : dict[str] -> list[str]
        Map variable names to their coordinates
    pred_dims: dict
        Dims for predictions data. Map variable names to their coordinates.
    num_chains: int
        Number of chains used for sampling. Ignored if posterior is present.
    """
    return NumPyroConverter(posterior=posterior, prior=prior, posterior_predictive=posterior_predictive, predictions=predictions, constant_data=constant_data, predictions_constant_data=predictions_constant_data, log_likelihood=log_likelihood, index_origin=index_origin, coords=coords, dims=dims, pred_dims=pred_dims, num_chains=num_chains).to_inference_data()