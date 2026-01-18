import contextlib
import gc
import operator
import os
import platform
import pprint
import re
import shutil
import sys
import warnings
from functools import wraps
from io import StringIO
from tempfile import mkdtemp, mkstemp
from warnings import WarningMessage
import torch._numpy as np
from torch._numpy import arange, asarray as asanyarray, empty, float32, intp, ndarray
import unittest
def assert_array_compare(comparison, x, y, err_msg='', verbose=True, header='', precision=6, equal_nan=True, equal_inf=True, *, strict=False):
    __tracebackhide__ = True
    from torch._numpy import all, array, asarray, bool_, inf, isnan, max
    x = asarray(x)
    y = asarray(y)

    def array2string(a):
        return str(a)
    ox, oy = (x, y)

    def func_assert_same_pos(x, y, func=isnan, hasval='nan'):
        """Handling nan/inf.

        Combine results of running func on x and y, checking that they are True
        at the same locations.

        """
        __tracebackhide__ = True
        x_id = func(x)
        y_id = func(y)
        if (x_id == y_id).all().item() is not True:
            msg = build_err_msg([x, y], err_msg + '\nx and y %s location mismatch:' % hasval, verbose=verbose, header=header, names=('x', 'y'), precision=precision)
            raise AssertionError(msg)
        if isinstance(x_id, bool) or x_id.ndim == 0:
            return bool_(x_id)
        elif isinstance(y_id, bool) or y_id.ndim == 0:
            return bool_(y_id)
        else:
            return y_id
    try:
        if strict:
            cond = x.shape == y.shape and x.dtype == y.dtype
        else:
            cond = (x.shape == () or y.shape == ()) or x.shape == y.shape
        if not cond:
            if x.shape != y.shape:
                reason = f'\n(shapes {x.shape}, {y.shape} mismatch)'
            else:
                reason = f'\n(dtypes {x.dtype}, {y.dtype} mismatch)'
            msg = build_err_msg([x, y], err_msg + reason, verbose=verbose, header=header, names=('x', 'y'), precision=precision)
            raise AssertionError(msg)
        flagged = bool_(False)
        if equal_nan:
            flagged = func_assert_same_pos(x, y, func=isnan, hasval='nan')
        if equal_inf:
            flagged |= func_assert_same_pos(x, y, func=lambda xy: xy == +inf, hasval='+inf')
            flagged |= func_assert_same_pos(x, y, func=lambda xy: xy == -inf, hasval='-inf')
        if flagged.ndim > 0:
            x, y = (x[~flagged], y[~flagged])
            if x.size == 0:
                return
        elif flagged:
            return
        val = comparison(x, y)
        if isinstance(val, bool):
            cond = val
            reduced = array([val])
        else:
            reduced = val.ravel()
            cond = reduced.all()
        if not cond:
            n_mismatch = reduced.size - int(reduced.sum(dtype=intp))
            n_elements = flagged.size if flagged.ndim != 0 else reduced.size
            percent_mismatch = 100 * n_mismatch / n_elements
            remarks = [f'Mismatched elements: {n_mismatch} / {n_elements} ({percent_mismatch:.3g}%)']
            with contextlib.suppress(TypeError, RuntimeError):
                error = abs(x - y)
                if np.issubdtype(x.dtype, np.unsignedinteger):
                    error2 = abs(y - x)
                    np.minimum(error, error2, out=error)
                max_abs_error = max(error)
                remarks.append('Max absolute difference: ' + array2string(max_abs_error.item()))
                nonzero = bool_(y != 0)
                if all(~nonzero):
                    max_rel_error = array(inf)
                else:
                    max_rel_error = max(error[nonzero] / abs(y[nonzero]))
                remarks.append('Max relative difference: ' + array2string(max_rel_error.item()))
            err_msg += '\n' + '\n'.join(remarks)
            msg = build_err_msg([ox, oy], err_msg, verbose=verbose, header=header, names=('x', 'y'), precision=precision)
            raise AssertionError(msg)
    except ValueError:
        import traceback
        efmt = traceback.format_exc()
        header = f'error during assertion:\n\n{efmt}\n\n{header}'
        msg = build_err_msg([x, y], err_msg, verbose=verbose, header=header, names=('x', 'y'), precision=precision)
        raise ValueError(msg)