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
def test_is_proxy():
    hdr = FunkyHeader((2, 3, 4))
    bio = BytesIO()
    prox = ArrayProxy(bio, hdr)
    assert is_proxy(prox)
    assert not is_proxy(bio)
    assert not is_proxy(hdr)
    assert not is_proxy(np.zeros((2, 3, 4)))

    class NP:
        is_proxy = False
    assert not is_proxy(NP())