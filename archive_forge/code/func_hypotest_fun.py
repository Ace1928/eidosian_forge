import numpy as np
from functools import wraps
from scipy._lib._docscrape import FunctionDoc, Parameter
from scipy._lib._util import _contains_nan, AxisError, _get_nan
import inspect
def hypotest_fun(x):
    samples = np.split(x, split_indices)[:n_samp + n_kwd_samp]
    if sentinel:
        samples = _remove_sentinel(samples, paired, sentinel)
    if is_too_small(samples, kwds):
        return np.full(n_out, NaN)
    return result_to_tuple(hypotest_fun_out(*samples, **kwds))