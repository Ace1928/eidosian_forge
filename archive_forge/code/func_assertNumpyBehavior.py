import sys
import os
import shutil
import inspect
import tempfile
import subprocess
from contextlib import contextmanager
from functools import wraps
import numpy as np
from numpy.lib.recfunctions import repack_fields
import h5py
import unittest as ut
def assertNumpyBehavior(self, dset, arr, s, skip_fast_reader=False):
    """ Apply slicing arguments "s" to both dset and arr.

        Succeeds if the results of the slicing are identical, or the
        exception raised is of the same type for both.

        "arr" must be a Numpy array; "dset" may be a NumPy array or dataset.
        """
    exc = None
    try:
        arr_result = arr[s]
    except Exception as e:
        exc = type(e)
    s_fast = s if isinstance(s, tuple) else (s,)
    if exc is None:
        self.assertArrayEqual(dset[s], arr_result)
        if not skip_fast_reader:
            self.assertArrayEqual(dset._fast_reader.read(s_fast), arr_result)
    else:
        with self.assertRaises(exc):
            dset[s]
        if not skip_fast_reader:
            with self.assertRaises(exc):
                dset._fast_reader.read(s_fast)