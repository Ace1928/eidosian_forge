from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def autograd_get_dtype_name(x):
    """A autograd version of get_dtype_name that can handle array boxes."""
    return ar.get_dtype_name(x._value)