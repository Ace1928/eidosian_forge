import os
from os.path import join as pjoin, dirname
import shutil
import tempfile
import warnings
from io import BytesIO
from glob import glob
from contextlib import contextmanager
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.io import netcdf_file
from scipy._lib._tmpdirs import in_tempdir
def assert_mask_matches(arr, expected_mask):
    """
    Asserts that the mask of arr is effectively the same as expected_mask.

    In contrast to numpy.ma.testutils.assert_mask_equal, this function allows
    testing the 'mask' of a standard numpy array (the mask in this case is treated
    as all False).

    Parameters
    ----------
    arr : ndarray or MaskedArray
        Array to test.
    expected_mask : array_like of booleans
        A list giving the expected mask.
    """
    mask = np.ma.getmaskarray(arr)
    assert_equal(mask, expected_mask)