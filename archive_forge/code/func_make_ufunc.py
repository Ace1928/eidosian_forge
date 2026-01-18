import warnings
from collections.abc import Sequence
from copy import copy as _copy
from copy import deepcopy as _deepcopy
import numpy as np
import pandas as pd
from scipy.fftpack import next_fast_len
from scipy.interpolate import CubicSpline
from scipy.stats.mstats import mquantiles
from xarray import apply_ufunc
from .. import _log
from ..utils import conditional_jit, conditional_vect, conditional_dask
from .density_utils import histogram as _histogram
def make_ufunc(func, n_dims=2, n_output=1, n_input=1, index=Ellipsis, ravel=True, check_shape=None):
    """Make ufunc from a function taking 1D array input.

    Parameters
    ----------
    func : callable
    n_dims : int, optional
        Number of core dimensions not broadcasted. Dimensions are skipped from the end.
        At minimum n_dims > 0.
    n_output : int, optional
        Select number of results returned by `func`.
        If n_output > 1, ufunc returns a tuple of objects else returns an object.
    n_input : int, optional
        Number of **array** inputs to func, i.e. ``n_input=2`` means that func is called
        with ``func(ary1, ary2, *args, **kwargs)``
    index : int, optional
        Slice ndarray with `index`. Defaults to `Ellipsis`.
    ravel : bool, optional
        If true, ravel the ndarray before calling `func`.
    check_shape: bool, optional
        If false, do not check if the shape of the output is compatible with n_dims and
        n_output. By default, True only for n_input=1. If n_input is larger than 1, the last
        input array is used to check the shape, however, shape checking with multiple inputs
        may not be correct.

    Returns
    -------
    callable
        ufunc wrapper for `func`.
    """
    if n_dims < 1:
        raise TypeError('n_dims must be one or higher.')
    if n_input == 1 and check_shape is None:
        check_shape = True
    elif check_shape is None:
        check_shape = False

    def _ufunc(*args, out=None, out_shape=None, **kwargs):
        """General ufunc for single-output function."""
        arys = args[:n_input]
        n_dims_out = None
        if out is None:
            if out_shape is None:
                out = np.empty(arys[-1].shape[:-n_dims])
            else:
                out = np.empty((*arys[-1].shape[:-n_dims], *out_shape))
                n_dims_out = -len(out_shape)
        elif check_shape:
            if out.shape != arys[-1].shape[:-n_dims]:
                msg = f'Shape incorrect for `out`: {out.shape}.'
                msg += f' Correct shape is {arys[-1].shape[:-n_dims]}'
                raise TypeError(msg)
        for idx in np.ndindex(out.shape[:n_dims_out]):
            arys_idx = [ary[idx].ravel() if ravel else ary[idx] for ary in arys]
            out_idx = np.asarray(func(*arys_idx, *args[n_input:], **kwargs))[index]
            if n_dims_out is None:
                out_idx = out_idx.item()
            out[idx] = out_idx
        return out

    def _multi_ufunc(*args, out=None, out_shape=None, **kwargs):
        """General ufunc for multi-output function."""
        arys = args[:n_input]
        element_shape = arys[-1].shape[:-n_dims]
        if out is None:
            if out_shape is None:
                out = tuple((np.empty(element_shape) for _ in range(n_output)))
            else:
                out = tuple((np.empty((*element_shape, *out_shape[i])) for i in range(n_output)))
        elif check_shape:
            raise_error = False
            correct_shape = tuple((element_shape for _ in range(n_output)))
            if isinstance(out, tuple):
                out_shape = tuple((item.shape for item in out))
                if out_shape != correct_shape:
                    raise_error = True
            else:
                raise_error = True
                out_shape = 'not tuple, type={type(out)}'
            if raise_error:
                msg = f'Shapes incorrect for `out`: {out_shape}.'
                msg += f' Correct shapes are {correct_shape}'
                raise TypeError(msg)
        for idx in np.ndindex(element_shape):
            arys_idx = [ary[idx].ravel() if ravel else ary[idx] for ary in arys]
            results = func(*arys_idx, *args[n_input:], **kwargs)
            for i, res in enumerate(results):
                out[i][idx] = np.asarray(res)[index]
        return out
    if n_output > 1:
        ufunc = _multi_ufunc
    else:
        ufunc = _ufunc
    update_docstring(ufunc, func, n_output)
    return ufunc