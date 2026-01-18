import itertools
import math
from functools import wraps
import numpy
import scipy.special as special
from .._config import get_config
from .fixes import parse_version
def supported_float_dtypes(xp):
    """Supported floating point types for the namespace

    Note: float16 is not officially part of the Array API spec at the
    time of writing but scikit-learn estimators and functions can choose
    to accept it when xp.float16 is defined.

    https://data-apis.org/array-api/latest/API_specification/data_types.html
    """
    if hasattr(xp, 'float16'):
        return (xp.float64, xp.float32, xp.float16)
    else:
        return (xp.float64, xp.float32)