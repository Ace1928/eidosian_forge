import warnings
import numpy as np
from ..utils import _var_names
from .converters import convert_to_dataset
def extract_dataset(data, group='posterior', combined=True, var_names=None, filter_vars=None, num_samples=None, rng=None):
    """Extract an InferenceData group or subset of it.

    .. deprecated:: 0.13
            `extract_dataset` will be removed in ArviZ 0.14, it is replaced by
            `extract` because the latter allows to obtain both DataSets and DataArrays.
    """
    warnings.warn('extract_dataset has been deprecated, please use extract', FutureWarning, stacklevel=2)
    data = extract(data=data, group=group, combined=combined, var_names=var_names, filter_vars=filter_vars, num_samples=num_samples, rng=rng)
    return data