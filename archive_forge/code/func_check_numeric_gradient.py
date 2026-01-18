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
def check_numeric_gradient(sym, location, aux_states=None, numeric_eps=None, rtol=None, atol=None, grad_nodes=None, use_forward_train=True, ctx=None, grad_stype_dict=None, dtype=default_dtype()):
    """Verify an operation by checking backward pass via finite difference method.

    Based on Theano's `theano.gradient.verify_grad` [1]

    Parameters
    ----------
    sym : Symbol
        Symbol containing op to test
    location : list or tuple or dict
        Argument values used as location to compute gradient

        - if type is list of numpy.ndarray,             inner elements should have the same order as mxnet.sym.list_arguments().

        - if type is dict of str -> numpy.ndarray,             maps the name of arguments to the corresponding numpy.ndarray.

        *In either case, value of all the arguments must be provided.*
    aux_states : list or tuple or dict, optional
        The auxiliary states required when generating the executor for the symbol.
    numeric_eps : float, optional
        Delta for the finite difference method that approximates the gradient.
    rtol : None or float
        The relative threshold. Default threshold will be used if set to ``None``.
    atol : None or float
        The absolute threshold. Default threshold will be used if set to ``None``.
    grad_nodes : None or list or tuple or dict, optional
        Names of the nodes to check gradient on
    use_forward_train : bool
        Whether to use is_train=True when computing the finite-difference.
    ctx : Context, optional
        Check the gradient computation on the specified device.
    grad_stype_dict : dict of str->str, optional
        Storage type dictionary for gradient ndarrays.
    dtype: np.float16 or np.float32 or np.float64
        Datatype for mx.nd.array.

    References
    ---------
    [1] https://github.com/Theano/Theano/blob/master/theano/gradient.py
    """
    assert dtype in (np.float16, np.float32, np.float64)
    if ctx is None:
        ctx = default_context()

    def random_projection(shape):
        """Get a random weight matrix with not too small elements

        Parameters
        ----------
        shape : list or tuple
        """
        plain = np.random.rand(*shape) + 0.1
        return plain
    location = _parse_location(sym=sym, location=location, ctx=ctx, dtype=dtype)
    location_npy = {k: v.asnumpy() for k, v in location.items()}
    aux_states = _parse_aux_states(sym=sym, aux_states=aux_states, ctx=ctx, dtype=dtype)
    if aux_states is not None:
        aux_states_npy = {k: v.asnumpy() for k, v in aux_states.items()}
    else:
        aux_states_npy = None
    if grad_nodes is None:
        grad_nodes = sym.list_arguments()
        grad_req = {k: 'write' for k in grad_nodes}
    elif isinstance(grad_nodes, (list, tuple)):
        grad_nodes = list(grad_nodes)
        grad_req = {k: 'write' for k in grad_nodes}
    elif isinstance(grad_nodes, dict):
        grad_req = grad_nodes.copy()
        grad_nodes = grad_nodes.keys()
    else:
        raise ValueError
    input_shape = {k: v.shape for k, v in location.items()}
    _, out_shape, _ = sym.infer_shape(**input_shape)
    proj = mx.sym.Variable('__random_proj')
    is_np_sym = bool(isinstance(sym, np_symbol))
    if is_np_sym:
        proj = proj.as_np_ndarray()
    out = sym * proj
    if is_np_sym:
        out = out.as_nd_ndarray()
    out = mx.sym.make_loss(out)
    location = dict(list(location.items()) + [('__random_proj', mx.nd.array(random_projection(out_shape[0]), ctx=ctx, dtype=dtype))])
    args_grad_npy = dict([(k, np.random.normal(0, 0.01, size=location[k].shape)) for k in grad_nodes] + [('__random_proj', np.random.normal(0, 0.01, size=out_shape[0]))])
    args_grad = {k: mx.nd.array(v, ctx=ctx, dtype=dtype) for k, v in args_grad_npy.items()}
    if grad_stype_dict is not None:
        assert isinstance(grad_stype_dict, dict), 'grad_stype_dict must be a dict'
        for k, v in grad_stype_dict.items():
            if k in args_grad and v in _STORAGE_TYPE_STR_TO_ID and (v != 'default'):
                args_grad[k] = mx.nd.zeros(args_grad[k].shape, args_grad[k].context, args_grad[k].dtype, v)
    executor = out.bind(ctx, grad_req=grad_req, args=location, args_grad=args_grad, aux_states=aux_states)
    inps = executor.arg_arrays
    if len(inps) != len(location):
        raise ValueError('Executor arg_arrays and and location len do not match.Got %d inputs and %d locations' % (len(inps), len(location)))
    assert len(executor.outputs) == 1
    executor.forward(is_train=True)
    eps = get_tolerance(executor.outputs[0], numeric_eps, default_numeric_eps())
    if dtype in (np.float32, np.float16):
        assert eps >= 1e-05
    executor.backward()
    symbolic_grads = executor.grad_dict
    numeric_gradients = numeric_grad(executor, location_npy, aux_states_npy, eps=eps, use_forward_train=use_forward_train, dtype=dtype)
    for name in grad_nodes:
        fd_grad = numeric_gradients[name]
        orig_grad = args_grad_npy[name]
        sym_grad = symbolic_grads[name]
        if grad_req[name] == 'write':
            assert_almost_equal(fd_grad, sym_grad, rtol, atol, ('NUMERICAL_%s' % name, 'BACKWARD_%s' % name))
        elif grad_req[name] == 'add':
            if isinstance(sym_grad, mx.nd.NDArray):
                sym_grad = sym_grad.asnumpy()
            assert_almost_equal(fd_grad, sym_grad - orig_grad, rtol, atol, ('NUMERICAL_%s' % name, 'BACKWARD_%s' % name))
        elif grad_req[name] == 'null':
            assert_almost_equal(orig_grad, sym_grad, rtol, atol, ('NUMERICAL_%s' % name, 'BACKWARD_%s' % name))
        else:
            raise ValueError('Invalid grad_req %s for argument %s' % (grad_req[name], name))