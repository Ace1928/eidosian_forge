from . import utils
from scipy import sparse
import numpy as np
import warnings
def arcsinh_transform(*args, **kwargs):
    warnings.warn('scprep.transform.arcsinh_transform is deprecated. Please use scprep.transform.arcsinh in future.', FutureWarning)
    return arcsinh(*args, **kwargs)