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
def shuffle_csr_column_indices(csr):
    """Shuffle CSR column indices per row
    This allows validation of unordered column indices, which is not a requirement
    for a valid CSR matrix
    """
    row_count = len(csr.indptr) - 1
    for i in range(row_count):
        start_index = csr.indptr[i]
        end_index = csr.indptr[i + 1]
        sublist = np.array(csr.indices[start_index:end_index])
        np.random.shuffle(sublist)
        csr.indices[start_index:end_index] = sublist