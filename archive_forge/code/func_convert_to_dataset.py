import numpy as np
import tree
import xarray as xr
from .base import dict_to_dataset
from .inference_data import InferenceData
from .io_beanmachine import from_beanmachine
from .io_cmdstan import from_cmdstan
from .io_cmdstanpy import from_cmdstanpy
from .io_emcee import from_emcee
from .io_numpyro import from_numpyro
from .io_pyro import from_pyro
from .io_pystan import from_pystan
def convert_to_dataset(obj, *, group='posterior', coords=None, dims=None):
    """Convert a supported object to an xarray dataset.

    This function is idempotent, in that it will return xarray.Dataset functions
    unchanged. Raises `ValueError` if the desired group can not be extracted.

    Note this goes through a DataInference object. See `convert_to_inference_data`
    for more details. Raises ValueError if it can not work out the desired
    conversion.

    Parameters
    ----------
    obj : dict, str, np.ndarray, xr.Dataset, pystan fit
        A supported object to convert to InferenceData:

        - InferenceData: returns unchanged
        - str: Attempts to load the netcdf dataset from disk
        - pystan fit: Automatically extracts data
        - xarray.Dataset: adds to InferenceData as only group
        - xarray.DataArray: creates an xarray dataset as the only group, gives the
          array an arbitrary name, if name not set
        - dict: creates an xarray dataset as the only group
        - numpy array: creates an xarray dataset as the only group, gives the
          array an arbitrary name

    group : str
        If `obj` is a dict or numpy array, assigns the resulting xarray
        dataset to this group.
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict[str, List(str)]
        A mapping from variables to a list of coordinate names for the variable

    Returns
    -------
    xarray.Dataset
    """
    inference_data = convert_to_inference_data(obj, group=group, coords=coords, dims=dims)
    dataset = getattr(inference_data, group, None)
    if dataset is None:
        raise ValueError('Can not extract {group} from {obj}! See {filename} for other conversion utilities.'.format(group=group, obj=obj, filename=__file__))
    return dataset