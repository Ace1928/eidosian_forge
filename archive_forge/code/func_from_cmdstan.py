import logging
import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from .. import utils
from ..rcparams import rcParams
from .base import CoordSpec, DimSpec, dict_to_dataset, infer_stan_dtypes, requires
from .inference_data import InferenceData
def from_cmdstan(posterior: Optional[Union[str, List[str]]]=None, *, posterior_predictive: Optional[Union[str, List[str]]]=None, predictions: Optional[Union[str, List[str]]]=None, prior: Optional[Union[str, List[str]]]=None, prior_predictive: Optional[Union[str, List[str]]]=None, observed_data: Optional[str]=None, observed_data_var: Optional[Union[str, List[str]]]=None, constant_data: Optional[str]=None, constant_data_var: Optional[Union[str, List[str]]]=None, predictions_constant_data: Optional[str]=None, predictions_constant_data_var: Optional[Union[str, List[str]]]=None, log_likelihood: Optional[Union[str, List[str]]]=None, index_origin: Optional[int]=None, coords: Optional[CoordSpec]=None, dims: Optional[DimSpec]=None, disable_glob: Optional[bool]=False, save_warmup: Optional[bool]=None, dtypes: Optional[Dict]=None) -> InferenceData:
    """Convert CmdStan data into an InferenceData object.

    For a usage example read the
    :ref:`Creating InferenceData section on from_cmdstan <creating_InferenceData>`

    Parameters
    ----------
    posterior : str or list of str, optional
        List of paths to output.csv files.
    posterior_predictive : str or list of str, optional
        Posterior predictive samples for the fit. If endswith ".csv" assumes file.
    predictions : str or list of str, optional
        Out of sample predictions samples for the fit. If endswith ".csv" assumes file.
    prior : str or list of str, optional
        List of paths to output.csv files
    prior_predictive : str or list of str, optional
        Prior predictive samples for the fit. If endswith ".csv" assumes file.
    observed_data : str, optional
        Observed data used in the sampling. Path to data file in Rdump or JSON format.
    observed_data_var : str or list of str, optional
        Variable(s) used for slicing observed_data. If not defined, all
        data variables are imported.
    constant_data : str, optional
        Constant data used in the sampling. Path to data file in Rdump or JSON format.
    constant_data_var : str or list of str, optional
        Variable(s) used for slicing constant_data. If not defined, all
        data variables are imported.
    predictions_constant_data : str, optional
        Constant data for predictions used in the sampling.
        Path to data file in Rdump or JSON format.
    predictions_constant_data_var : str or list of str, optional
        Variable(s) used for slicing predictions_constant_data.
        If not defined, all data variables are imported.
    log_likelihood : dict of {str: str}, list of str or str, optional
        Pointwise log_likelihood for the data. log_likelihood is extracted from the
        posterior. It is recommended to use this argument as a dictionary whose keys
        are observed variable names and its values are the variables storing log
        likelihood arrays in the Stan code. In other cases, a dictionary with keys
        equal to its values is used. By default, if a variable ``log_lik`` is
        present in the Stan model, it will be retrieved as pointwise log
        likelihood values. Use ``False`` to avoid this behaviour.
    index_origin : int, optional
        Starting value of integer coordinate values. Defaults to the value in rcParam
        ``data.index_origin``.
    coords : dict of {str: array_like}, optional
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict of {str: list of str}, optional
        A mapping from variables to a list of coordinate names for the variable.
    disable_glob : bool
        Don't use glob for string input. This means that all string input is
        assumed to be variable names (samples) or a path (data).
    save_warmup : bool
        Save warmup iterations into InferenceData object, if found in the input files.
        If not defined, use default defined by the rcParams.
    dtypes : dict or str
        A dictionary containing dtype information (int, float) for parameters.
        If input is a string, it is assumed to be a model code or path to model code file.

    Returns
    -------
    InferenceData object
    """
    return CmdStanConverter(posterior=posterior, posterior_predictive=posterior_predictive, predictions=predictions, prior=prior, prior_predictive=prior_predictive, observed_data=observed_data, observed_data_var=observed_data_var, constant_data=constant_data, constant_data_var=constant_data_var, predictions_constant_data=predictions_constant_data, predictions_constant_data_var=predictions_constant_data_var, log_likelihood=log_likelihood, index_origin=index_origin, coords=coords, dims=dims, disable_glob=disable_glob, save_warmup=save_warmup, dtypes=dtypes).to_inference_data()