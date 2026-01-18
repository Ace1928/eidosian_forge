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
def gen_buckets_probs_with_ppf(ppf, nbuckets):
    """Generate the buckets and probabilities for chi_square test when the ppf (Quantile function)
     is specified.

    Parameters
    ----------
    ppf : function
        The Quantile function that takes a probability and maps it back to a value.
        It's the inverse of the cdf function
    nbuckets : int
        size of the buckets

    Returns
    -------
    buckets : list of tuple
        The generated buckets
    probs : list
        The generate probabilities
    """
    assert nbuckets > 0
    probs = [1.0 / nbuckets for _ in range(nbuckets)]
    buckets = [(ppf(i / float(nbuckets)), ppf((i + 1) / float(nbuckets))) for i in range(nbuckets)]
    return (buckets, probs)