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
def assign_each2(input1, input2, function):
    """Return ndarray composed of passing two array values through some function"""
    if function is None:
        output = np.array(input1)
    else:
        assert input1.shape == input2.shape
        it_input1 = np.nditer(input1, flags=['f_index'])
        it_input2 = np.nditer(input2, flags=['f_index'])
        output = np.zeros(input1.shape)
        it_out = np.nditer(output, flags=['f_index'], op_flags=['writeonly'])
        while not it_input1.finished:
            val_input1 = it_input1[0]
            val_input2 = it_input2[0]
            it_out[0] = function(val_input1, val_input2)
            it_input1.iternext()
            it_input2.iternext()
            it_out.iternext()
    return output