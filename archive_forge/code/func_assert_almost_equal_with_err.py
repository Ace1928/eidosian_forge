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
def assert_almost_equal_with_err(a, b, rtol=None, atol=None, etol=None, names=('a', 'b'), equal_nan=False, mismatches=(10, 10)):
    """Test that two numpy arrays are almost equal within given error rate. Raise exception message if not.

    Parameters
    ----------
    a : np.ndarray
    b : np.ndarray
    rtol : None or float or dict of dtype -> float
        The relative threshold. Default threshold will be used if set to ``None``.
    atol : None or float or dict of dtype -> float
        The absolute threshold. Default threshold will be used if set to ``None``.
    threshold : None or float
        The checking threshold. Default threshold will be used if set to ``None``.
    etol : None or float
        The error rate threshold. If etol is float, return true if error_rate < etol even if
        any error is found.
    names : tuple of names, optional
        The names used in error message when an exception occurs
    equal_nan : boolean, optional
        The flag determining how to treat NAN values in comparison
    mismatches : tuple of mismatches
        Maximum number of mismatches to be printed (mismatches[0]) and determine (mismatches[1])
    """
    etol = get_etol(etol)
    if etol > 0:
        rtol, atol = get_tols(a, b, rtol, atol)
        if isinstance(a, mx.nd.NDArray):
            a = a.asnumpy()
        if isinstance(b, mx.nd.NDArray):
            b = b.asnumpy()
        equals = np.isclose(a, b, rtol=rtol, atol=atol)
        err = 1 - np.count_nonzero(equals) / equals.size
        if err > etol:
            index, rel = _find_max_violation(a, b, rtol, atol)
            indexErr = index
            relErr = rel
            print('\n*** Maximum errors for vector of size {}:  rtol={}, atol={}\n'.format(a.size, rtol, atol))
            aTmp = a.copy()
            bTmp = b.copy()
            i = 1
            while i <= a.size:
                if i <= mismatches[0]:
                    print('%3d: Error %f  %s' % (i, rel, locationError(a, b, index, names)))
                aTmp[index] = bTmp[index] = 0
                if almost_equal(aTmp, bTmp, rtol, atol, equal_nan=equal_nan):
                    break
                i += 1
                if i <= mismatches[1] or mismatches[1] <= 0:
                    index, rel = _find_max_violation(aTmp, bTmp, rtol, atol)
                else:
                    break
            mismatchDegree = 'at least ' if mismatches[1] > 0 and i > mismatches[1] else ''
            errMsg = 'Error %f exceeds tolerance rtol=%e, atol=%e (mismatch %s%f%%).\n%s' % (relErr, rtol, atol, mismatchDegree, 100 * i / a.size, locationError(a, b, indexErr, names, maxError=True))
            np.set_printoptions(threshold=4, suppress=True)
            msg = npt.build_err_msg([a, b], err_msg=errMsg)
            raise AssertionError(msg)
    else:
        assert_almost_equal(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)