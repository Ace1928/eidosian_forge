import itertools
import unittest
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..optpkg import optional_package
from ..casting import sctypes_aliases, shared_range, type_info
from ..spatialimages import HeaderDataError
from ..spm99analyze import HeaderTypeError, Spm99AnalyzeHeader, Spm99AnalyzeImage
from ..testing import (
from ..volumeutils import _dt_min_max, apply_read_scaling
from . import test_analyze
def test_data_scaling(self):
    hdr = self.header_class()
    hdr.set_data_shape((1, 2, 3))
    hdr.set_data_dtype(np.int16)
    S3 = BytesIO()
    data = np.arange(6, dtype=np.float64).reshape((1, 2, 3))
    hdr.data_to_fileobj(data, S3)
    data_back = hdr.data_from_fileobj(S3)
    assert_array_almost_equal(data, data_back, 4)
    assert not np.all(data == data_back)
    data_back2 = hdr.data_from_fileobj(S3)
    assert_array_equal(data_back, data_back2, 4)
    hdr.data_to_fileobj(data, S3, rescale=True)
    data_back = hdr.data_from_fileobj(S3)
    assert_array_almost_equal(data, data_back, 4)
    assert not np.all(data == data_back)
    with np.errstate(invalid='ignore'):
        hdr.data_to_fileobj(data, S3, rescale=False)
    data_back = hdr.data_from_fileobj(S3)
    assert np.all(data == data_back)