import gzip
from copy import copy
from decimal import Decimal
from hashlib import sha1
from os.path import dirname
from os.path import join as pjoin
from unittest import TestCase
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ...tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from ...volumeutils import endian_codes
from .. import dicomreaders as didr
from .. import dicomwrappers as didw
from . import dicom_test, have_dicom, pydicom
@dicom_test
def test_data_fake(self):
    fake_mf = copy(self.MINIMAL_MF)
    MFW = self.WRAPCLASS
    dw = MFW(fake_mf)
    with pytest.raises(didw.WrapperError):
        dw.get_data()
    dw.image_shape = (2, 3, 4)
    with pytest.raises(didw.WrapperError):
        dw.get_data()
    fake_mf['Rows'] = 2
    fake_mf['Columns'] = 3
    dim_idxs = ((1, 1), (1, 2), (1, 3), (1, 4))
    fake_mf.update(fake_shape_dependents(dim_idxs, sid_dim=0))
    assert MFW(fake_mf).image_shape == (2, 3, 4)
    with pytest.raises(didw.WrapperError):
        dw.get_data()
    data = np.arange(24).reshape((2, 3, 4))
    fake_mf['pixel_array'] = np.rollaxis(data, 2)
    dw = MFW(fake_mf)
    assert_array_equal(dw.get_data(), data)
    fake_mf['RescaleSlope'] = 2.0
    fake_mf['RescaleIntercept'] = -1
    assert_array_equal(MFW(fake_mf).get_data(), data * 2.0 - 1)
    dim_idxs = ((1, 4), (1, 2), (1, 3), (1, 1))
    fake_mf.update(fake_shape_dependents(dim_idxs, sid_dim=0))
    sorted_data = data[..., [3, 1, 2, 0]]
    fake_mf['pixel_array'] = np.rollaxis(sorted_data, 2)
    assert_array_equal(MFW(fake_mf).get_data(), data * 2.0 - 1)
    dim_idxs = [[1, 4, 2, 1], [1, 2, 2, 1], [1, 3, 2, 1], [1, 1, 2, 1], [1, 4, 2, 2], [1, 2, 2, 2], [1, 3, 2, 2], [1, 1, 2, 2], [1, 4, 1, 1], [1, 2, 1, 1], [1, 3, 1, 1], [1, 1, 1, 1], [1, 4, 1, 2], [1, 2, 1, 2], [1, 3, 1, 2], [1, 1, 1, 2]]
    fake_mf.update(fake_shape_dependents(dim_idxs, sid_dim=0))
    shape = (2, 3, 4, 2, 2)
    data = np.arange(np.prod(shape)).reshape(shape)
    sorted_data = data.reshape(shape[:2] + (-1,), order='F')
    order = [11, 9, 10, 8, 3, 1, 2, 0, 15, 13, 14, 12, 7, 5, 6, 4]
    sorted_data = sorted_data[..., np.argsort(order)]
    fake_mf['pixel_array'] = np.rollaxis(sorted_data, 2)
    assert_array_equal(MFW(fake_mf).get_data(), data * 2.0 - 1)