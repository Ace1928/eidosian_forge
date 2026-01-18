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
def assertArrayEqual(self, dset, arr, message=None, precision=None, check_alignment=True):
    """ Make sure dset and arr have the same shape, dtype and contents, to
            within the given precision, optionally ignoring differences in dtype alignment.

            Note that dset may be a NumPy array or an HDF5 dataset.
        """
    if precision is None:
        precision = 1e-05
    if message is None:
        message = ''
    else:
        message = ' (%s)' % message
    if np.isscalar(dset) or np.isscalar(arr):
        assert np.isscalar(dset) and np.isscalar(arr), 'Scalar/array mismatch ("%r" vs "%r")%s' % (dset, arr, message)
        dset = np.asarray(dset)
        arr = np.asarray(arr)
    assert dset.shape == arr.shape, 'Shape mismatch (%s vs %s)%s' % (dset.shape, arr.shape, message)
    if dset.dtype != arr.dtype:
        if check_alignment:
            normalized_dset_dtype = dset.dtype
            normalized_arr_dtype = arr.dtype
        else:
            normalized_dset_dtype = repack_fields(dset.dtype)
            normalized_arr_dtype = repack_fields(arr.dtype)
        assert normalized_dset_dtype == normalized_arr_dtype, 'Dtype mismatch (%s vs %s)%s' % (normalized_dset_dtype, normalized_dset_dtype, message)
        if not check_alignment:
            if normalized_dset_dtype != dset.dtype:
                dset = repack_fields(np.asarray(dset))
            if normalized_arr_dtype != arr.dtype:
                arr = repack_fields(np.asarray(arr))
    if arr.dtype.names is not None:
        for n in arr.dtype.names:
            message = '[FIELD %s] %s' % (n, message)
            self.assertArrayEqual(dset[n], arr[n], message=message, precision=precision, check_alignment=check_alignment)
    elif arr.dtype.kind in ('i', 'f'):
        assert np.all(np.abs(dset[...] - arr[...]) < precision), 'Arrays differ by more than %.3f%s' % (precision, message)
    elif arr.dtype.kind == 'O':
        for v1, v2 in zip(dset.flat, arr.flat):
            self.assertArrayEqual(v1, v2, message=message, precision=precision, check_alignment=check_alignment)
    else:
        assert np.all(dset[...] == arr[...]), 'Arrays are not equal (dtype %s) %s' % (arr.dtype.str, message)