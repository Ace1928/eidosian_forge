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
def test_keep_file_open_true_false_invalid():
    tests = [('open', False, False, False, False), ('open', False, True, False, False), ('open', True, False, False, False), ('open', True, True, False, False), ('bin', False, False, False, False), ('bin', False, True, False, False), ('bin', True, False, True, True), ('bin', True, True, True, True), ('gz', False, False, False, False), ('gz', False, True, True, False), ('gz', True, False, True, True), ('gz', True, True, True, True)]
    dtype = np.float32
    data = np.arange(1000, dtype=dtype).reshape((10, 10, 10))
    voxels = np.random.randint(0, 10, (10, 3))
    for test in tests:
        filetype, kfo, have_igzip, exp_persist, exp_kfo = test
        with InTemporaryDirectory(), mock.patch('nibabel.openers.ImageOpener', CountingImageOpener), patch_indexed_gzip(have_igzip):
            fname = f'testdata.{filetype}'
            if filetype == 'gz':
                with gzip.open(fname, 'wb') as fobj:
                    fobj.write(data.tobytes(order='F'))
            else:
                with open(fname, 'wb') as fobj:
                    fobj.write(data.tobytes(order='F'))
            if filetype == 'open':
                fobj1 = open(fname, 'rb')
                fobj2 = open(fname, 'rb')
            else:
                fobj1 = fname
                fobj2 = fname
            try:
                proxy = ArrayProxy(fobj1, ((10, 10, 10), dtype), keep_file_open=kfo)
                with patch_keep_file_open_default(kfo):
                    proxy_def = ArrayProxy(fobj2, ((10, 10, 10), dtype))
                assert proxy._persist_opener == exp_persist
                assert proxy._keep_file_open == exp_kfo
                assert proxy_def._persist_opener == exp_persist
                assert proxy_def._keep_file_open == exp_kfo
                if exp_persist:
                    assert _count_ImageOpeners(proxy, data, voxels) == 1
                    assert _count_ImageOpeners(proxy_def, data, voxels) == 1
                else:
                    assert _count_ImageOpeners(proxy, data, voxels) == 10
                    assert _count_ImageOpeners(proxy_def, data, voxels) == 10
                if filetype == 'gz' and have_igzip:
                    assert proxy._opener.fobj._drop_handles == (not exp_kfo)
                if filetype == 'open':
                    assert not fobj1.closed
                    assert not fobj2.closed
            finally:
                del proxy
                del proxy_def
                if filetype == 'open':
                    fobj1.close()
                    fobj2.close()
    with InTemporaryDirectory():
        fname = 'testdata'
        with open(fname, 'wb') as fobj:
            fobj.write(data.tobytes(order='F'))
        for invalid_kfo in (55, 'auto', 'cauto'):
            with pytest.raises(ValueError):
                ArrayProxy(fname, ((10, 10, 10), dtype), keep_file_open=invalid_kfo)
            with patch_keep_file_open_default(invalid_kfo):
                with pytest.raises(ValueError):
                    ArrayProxy(fname, ((10, 10, 10), dtype))