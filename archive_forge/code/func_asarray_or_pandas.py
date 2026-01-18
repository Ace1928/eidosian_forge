import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def asarray_or_pandas(a, copy=False, dtype=None, subok=False):
    if have_pandas:
        if isinstance(a, (pandas.Series, pandas.DataFrame)):
            extra_args = {}
            if hasattr(a, 'name'):
                extra_args['name'] = a.name
            return a.__class__(a, copy=copy, dtype=dtype, **extra_args)
    return np.array(a, copy=copy, dtype=dtype, subok=subok)