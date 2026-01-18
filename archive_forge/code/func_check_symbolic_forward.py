import time
import gzip
import struct
import traceback
import numbers
import sys
import os
import platform
import errno
import logging
import bz2
import zipfile
import json
from contextlib import contextmanager
from collections import OrderedDict
import numpy as np
import numpy.testing as npt
import numpy.random as rnd
import mxnet as mx
from .context import Context, current_context
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID
from .ndarray import array
from .symbol import Symbol
from .symbol.numpy import _Symbol as np_symbol
from .util import use_np, getenv, setenv  # pylint: disable=unused-import
from .runtime import Features
from .numpy_extension import get_cuda_compute_capability
def check_symbolic_forward(sym, location, expected, rtol=None, atol=None, aux_states=None, ctx=None, equal_nan=False, dtype=default_dtype()):
    """Compares a symbol's forward results with the expected ones.
    Prints error messages if the forward results are not the same as the expected ones.

    Parameters
    ---------
    sym : Symbol
        output symbol
    location : list of np.ndarray or dict of str to np.ndarray
        The evaluation point

        - if type is list of np.ndarray
            Contains all the numpy arrays corresponding to `sym.list_arguments()`.
        - if type is dict of str to np.ndarray
            Contains the mapping between argument names and their values.
    expected : list of np.ndarray or dict of str to np.ndarray
        The expected output value

        - if type is list of np.ndarray
            Contains arrays corresponding to exe.outputs.
        - if type is dict of str to np.ndarray
            Contains mapping between sym.list_output() and exe.outputs.
    rtol : None or float
        The relative threshold. Default threshold will be used if set to ``None``.
    atol : None or float
        The absolute threshold. Default threshold will be used if set to ``None``.
    aux_states : list of np.ndarray of dict, optional
        - if type is list of np.ndarray
            Contains all the NumPy arrays corresponding to sym.list_auxiliary_states
        - if type is dict of str to np.ndarray
            Contains the mapping between names of auxiliary states and their values.
    ctx : Context, optional
        running context
    dtype: "asnumpy" or np.float16 or np.float32 or np.float64
        If dtype is "asnumpy" then the mx.nd.array created will have the same
        type as th numpy array from which it is copied.
        Otherwise, dtype is the explicit datatype for all mx.nd.array objects
        created in this function.

    equal_nan: Boolean
        if True, `nan` is a valid value for checking equivalency (ie `nan` == `nan`)

    Example
    -------
    >>> shape = (2, 2)
    >>> lhs = mx.symbol.Variable('lhs')
    >>> rhs = mx.symbol.Variable('rhs')
    >>> sym_dot = mx.symbol.dot(lhs, rhs)
    >>> mat1 = np.array([[1, 2], [3, 4]])
    >>> mat2 = np.array([[5, 6], [7, 8]])
    >>> ret_expected = np.array([[19, 22], [43, 50]])
    >>> check_symbolic_forward(sym_dot, [mat1, mat2], [ret_expected])
    """
    assert dtype == 'asnumpy' or dtype in (np.float16, np.float32, np.float64)
    if ctx is None:
        ctx = default_context()
    location = _parse_location(sym=sym, location=location, ctx=ctx, dtype=dtype)
    aux_states = _parse_aux_states(sym=sym, aux_states=aux_states, ctx=ctx, dtype=dtype)
    if isinstance(expected, dict):
        expected = [expected[k] for k in sym.list_outputs()]
    args_grad_data = {k: mx.nd.empty(v.shape, ctx=ctx, dtype=v.dtype if dtype == 'asnumpy' else dtype) for k, v in location.items()}
    executor = sym.bind(ctx=ctx, args=location, args_grad=args_grad_data, aux_states=aux_states)
    for g in executor.grad_arrays:
        if g.ndim == 0:
            g[()] = 0
        else:
            g[:] = 0
    executor.forward(is_train=False)
    outputs = executor.outputs
    for output_name, expect, output in zip(sym.list_outputs(), expected, outputs):
        assert_almost_equal(expect, output, rtol, atol, ('EXPECTED_%s' % output_name, 'FORWARD_%s' % output_name), equal_nan=equal_nan)
    return executor.outputs