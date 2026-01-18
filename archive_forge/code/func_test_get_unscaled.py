import contextlib
import gzip
import pickle
from io import BytesIO
from unittest import mock
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from packaging.version import Version
from .. import __version__
from ..arrayproxy import ArrayProxy, get_obj_dtype, is_proxy, reshape_dataobj
from ..deprecator import ExpiredDeprecationError
from ..nifti1 import Nifti1Header, Nifti1Image
from ..openers import ImageOpener
from ..testing import memmap_after_ufunc
from ..tmpdirs import InTemporaryDirectory
from .test_fileslice import slicer_samples
from .test_openers import patch_indexed_gzip
def test_get_unscaled():

    class FunkyHeader2(FunkyHeader):

        def get_slope_inter(self):
            return (2.1, 3.14)
    shape = (2, 3, 4)
    hdr = FunkyHeader2(shape)
    bio = BytesIO()
    arr = np.arange(24, dtype=np.int32).reshape(shape, order='F')
    bio.write(b'\x00' * hdr.get_data_offset())
    bio.write(arr.tobytes(order='F'))
    prox = ArrayProxy(bio, hdr)
    assert_array_almost_equal(np.array(prox), arr * 2.1 + 3.14)
    assert_array_almost_equal(prox.get_unscaled(), arr)