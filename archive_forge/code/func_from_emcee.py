import warnings
from collections import OrderedDict
import numpy as np
import xarray as xr
from .. import utils
from .base import dict_to_dataset, generate_dims_coords, make_attrs
from .inference_data import InferenceData
def from_emcee(sampler=None, var_names=None, slices=None, arg_names=None, arg_groups=None, blob_names=None, blob_groups=None, index_origin=None, coords=None, dims=None):
    """Convert emcee data into an InferenceData object.

    For a usage example read :ref:`emcee_conversion`


    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        Fitted sampler from emcee.
    var_names : list of str, optional
        A list of names for variables in the sampler
    slices : list of array-like or slice, optional
        A list containing the indexes of each variable. Should only be used
        for multidimensional variables.
    arg_names : list of str, optional
        A list of names for args in the sampler
    arg_groups : list of str, optional
        A list of the group names (either ``observed_data`` or ``constant_data``) where
        args in the sampler are stored. If None, all args will be stored in observed
        data group.
    blob_names : list of str, optional
        A list of names for blobs in the sampler. When None,
        blobs are omitted, independently of them being present
        in the sampler or not.
    blob_groups : list of str, optional
        A list of the groups where blob_names variables
        should be assigned respectively. If blob_names!=None
        and blob_groups is None, all variables are assigned
        to log_likelihood group
    coords : dict of {str : array_like}, optional
        Map of dimensions to coordinates
    dims : dict of {str : list of str}, optional
        Map variable names to their coordinates

    Returns
    -------
    arviz.InferenceData

    """
    return EmceeConverter(sampler=sampler, var_names=var_names, slices=slices, arg_names=arg_names, arg_groups=arg_groups, blob_names=blob_names, blob_groups=blob_groups, index_origin=index_origin, coords=coords, dims=dims).to_inference_data()