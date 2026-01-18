import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def safe_is_pandas_categorical_dtype(dt):
    if not have_pandas_categorical_dtype:
        return False
    return _pandas_is_categorical_dtype(dt)