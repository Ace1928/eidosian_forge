import logging
from typing import Callable, Optional
import warnings
import numpy as np
from packaging import version
from .. import utils
from ..rcparams import rcParams
from .base import dict_to_dataset, requires
from .inference_data import InferenceData
def from_pyro(posterior=None, *, prior=None, posterior_predictive=None, log_likelihood=None, predictions=None, constant_data=None, predictions_constant_data=None, coords=None, dims=None, pred_dims=None, num_chains=1):
    """Convert Pyro data into an InferenceData object.

    For a usage example read the
    :ref:`Creating InferenceData section on from_pyro <creating_InferenceData>`


    Parameters
    ----------
    posterior : pyro.infer.MCMC
        Fitted MCMC object from Pyro
    prior: dict
        Prior samples from a Pyro model
    posterior_predictive : dict
        Posterior predictive samples for the posterior
    log_likelihood : bool, optional
        Calculate and store pointwise log likelihood values. Defaults to the value
        of rcParam ``data.log_likelihood``.
    predictions: dict
        Out of sample predictions
    constant_data: dict
        Dictionary containing constant data variables mapped to their values.
    predictions_constant_data: dict
        Constant data used for out-of-sample predictions.
    coords : dict[str] -> list[str]
        Map of dimensions to coordinates
    dims : dict[str] -> list[str]
        Map variable names to their coordinates
    pred_dims: dict
        Dims for predictions data. Map variable names to their coordinates.
    num_chains: int
        Number of chains used for sampling. Ignored if posterior is present.
    """
    return PyroConverter(posterior=posterior, prior=prior, posterior_predictive=posterior_predictive, log_likelihood=log_likelihood, predictions=predictions, constant_data=constant_data, predictions_constant_data=predictions_constant_data, coords=coords, dims=dims, pred_dims=pred_dims, num_chains=num_chains).to_inference_data()