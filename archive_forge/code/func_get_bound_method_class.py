import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
def get_bound_method_class(m):
    """Return the class for a bound method."""
    return m.im_class if sys.version < '3' else m.__self__.__class__