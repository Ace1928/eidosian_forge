import itertools
import sys
import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nibabel.tmpdirs import InTemporaryDirectory
from ... import load
from ...fileholders import FileHolder
from ...nifti1 import data_type_codes
from ...testing import get_test_data
from .. import (
from .test_parse_gifti_fast import (
def test_dataarray_init():
    gda = GiftiDataArray
    assert gda(None).data is None
    arr = np.arange(12, dtype=np.float32).reshape((3, 4))
    assert_array_equal(gda(arr).data, arr)
    pytest.raises(KeyError, gda, intent=1)
    pytest.raises(KeyError, gda, intent='not an intent')
    assert gda(intent=2).intent == 2
    assert gda(intent='correlation').intent == 2
    assert gda(intent='NIFTI_INTENT_CORREL').intent == 2
    assert gda(datatype=2).datatype == 2
    assert gda(datatype='uint8').datatype == 2
    pytest.raises(KeyError, gda, datatype='not_datatype')
    assert gda(arr).datatype == 16
    assert gda(arr, datatype='uint8').datatype == 2
    assert gda(encoding=1).encoding == 1
    assert gda(encoding='ASCII').encoding == 1
    assert gda(encoding='GIFTI_ENCODING_ASCII').encoding == 1
    pytest.raises(KeyError, gda, encoding='not an encoding')
    assert gda(endian=1).endian == 1
    assert gda(endian='big').endian == 1
    assert gda(endian='GIFTI_ENDIAN_BIG').endian == 1
    pytest.raises(KeyError, gda, endian='not endian code')
    aff = np.diag([2, 3, 4, 1])
    cs = GiftiCoordSystem(1, 2, aff)
    da = gda(coordsys=cs)
    assert da.coordsys.dataspace == 1
    assert da.coordsys.xformspace == 2
    assert_array_equal(da.coordsys.xform, aff)
    assert gda(ordering=2).ind_ord == 2
    assert gda(ordering='F').ind_ord == 2
    assert gda(ordering='ColumnMajorOrder').ind_ord == 2
    pytest.raises(KeyError, gda, ordering='not an ordering')
    meta_dict = dict(one=1, two=2)
    assert gda(meta=GiftiMetaData(meta_dict)).meta == meta_dict
    assert gda(meta=meta_dict).meta == meta_dict
    assert gda(meta=None).meta == {}
    assert gda(ext_fname='foo').ext_fname == 'foo'
    assert gda(ext_offset=12).ext_offset == 12