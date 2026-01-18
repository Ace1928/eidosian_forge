import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def safe_is_pandas_categorical(data):
    if not have_pandas_categorical:
        return False
    if isinstance(data, pandas.Categorical):
        return True
    if hasattr(data, 'dtype'):
        return safe_is_pandas_categorical_dtype(data.dtype)
    return False