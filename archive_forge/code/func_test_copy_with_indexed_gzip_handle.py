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
def test_copy_with_indexed_gzip_handle(tmp_path):
    indexed_gzip = pytest.importorskip('indexed_gzip')
    spec = ((50, 50, 50, 50), np.float32, 352, 1, 0)
    data = np.arange(np.prod(spec[0]), dtype=spec[1]).reshape(spec[0])
    fname = str(tmp_path / 'test.nii.gz')
    Nifti1Image(data, np.eye(4)).to_filename(fname)
    with indexed_gzip.IndexedGzipFile(fname) as fobj:
        proxy = ArrayProxy(fobj, spec)
        copied = proxy.copy()
        assert proxy.file_like is copied.file_like
        assert np.array_equal(proxy[0, 0, 0], copied[0, 0, 0])
        assert np.array_equal(proxy[-1, -1, -1], copied[-1, -1, -1])