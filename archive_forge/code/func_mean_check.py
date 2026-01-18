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
def mean_check(generator, mu, sigma, nsamples=1000000):
    """Test the generator by matching the mean.

    We test the sample mean by checking if it falls inside the range
        (mu - 3 * sigma / sqrt(n), mu + 3 * sigma / sqrt(n))

    References::

        @incollection{goucher2009beautiful,
              title={Beautiful Testing: Leading Professionals Reveal How They Improve Software},
              author={Goucher, Adam and Riley, Tim},
              year={2009},
              chapter=10
        }

    Examples::

        generator = lambda x: np.random.normal(0, 1.0, size=x)
        mean_check_ret = mean_check(generator, 0, 1.0)

    Parameters
    ----------
    generator : function
        The generator function. It's expected to generate N i.i.d samples by calling generator(N).
    mu : float
    sigma : float
    nsamples : int

    Returns
    -------
    ret : bool
        Whether the mean test succeeds
    """
    samples = np.array(generator(nsamples))
    sample_mean = samples.mean()
    ret = sample_mean > mu - 3 * sigma / np.sqrt(nsamples) and sample_mean < mu + 3 * sigma / np.sqrt(nsamples)
    return ret