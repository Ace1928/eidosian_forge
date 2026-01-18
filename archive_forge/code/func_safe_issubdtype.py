import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def safe_issubdtype(dt1, dt2):
    if safe_is_pandas_categorical_dtype(dt1):
        return False
    return np.issubdtype(dt1, dt2)