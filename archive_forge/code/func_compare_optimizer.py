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
def compare_optimizer(opt1, opt2, shape, dtype, w_stype='default', g_stype='default', rtol=0.0001, atol=1e-05, compare_states=True, ntensors=1):
    """Compare opt1 and opt2."""
    if not isinstance(shape, list):
        assert ntensors == 1
        if w_stype == 'default':
            w2 = mx.random.uniform(shape=shape, ctx=default_context(), dtype=dtype)
            w1 = w2.copyto(default_context())
        elif w_stype in ('row_sparse', 'csr'):
            w2 = rand_ndarray(shape, w_stype, density=1, dtype=dtype)
            w1 = w2.copyto(default_context()).tostype('default')
        else:
            raise Exception('type not supported yet')
        if g_stype == 'default':
            g2 = mx.random.uniform(shape=shape, ctx=default_context(), dtype=dtype)
            g1 = g2.copyto(default_context())
        elif g_stype in ('row_sparse', 'csr'):
            g2 = rand_ndarray(shape, g_stype, dtype=dtype)
            g1 = g2.copyto(default_context()).tostype('default')
        else:
            raise Exception('type not supported yet')
        state1 = opt1.create_state_multi_precision(0, w1)
        state2 = opt2.create_state_multi_precision(0, w2)
        if compare_states:
            compare_ndarray_tuple(state1, state2)
        opt1.update_multi_precision(0, w1, g1, state1)
        opt2.update_multi_precision(0, w2, g2, state2)
        if compare_states:
            compare_ndarray_tuple(state1, state2, rtol=rtol, atol=atol)
        assert_almost_equal(w1, w2, rtol=rtol, atol=atol)
    else:
        from copy import deepcopy
        w1, g1 = ([], [])
        for s in shape:
            w1.append(mx.random.uniform(shape=s, ctx=default_context(), dtype=dtype))
            g1.append(mx.random.uniform(shape=s, ctx=default_context(), dtype=dtype))
        w1 = tuple(w1)
        w2 = deepcopy(w1)
        g1 = tuple(g1)
        g2 = deepcopy(g1)
        state2 = [opt2.create_state_multi_precision(0, w2[i]) for i in range(ntensors)]
        opt2.update_multi_precision(list(range(ntensors)), w2, g2, state2)
        for i in range(ntensors):
            state1 = opt1.create_state_multi_precision(i, w1[i])
            opt1.update_multi_precision(i, w1[i], g1[i], state1)
            if compare_states:
                compare_ndarray_tuple(state1, state2[i], rtol, atol)
            compare_ndarray_tuple(w1[i], w2[i], rtol, atol)