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
def create_sparse_array_zd(shape, stype, density, data_init=None, rsp_indices=None, dtype=None, modifier_func=None, shuffle_csr_indices=False):
    """Create sparse array, using only rsp_indices to determine density"""
    if stype == 'row_sparse':
        density = 0.0
        if rsp_indices is not None:
            assert len(rsp_indices) <= shape[0]
    return create_sparse_array(shape, stype, data_init=data_init, rsp_indices=rsp_indices, dtype=dtype, modifier_func=modifier_func, density=density, shuffle_csr_indices=shuffle_csr_indices)