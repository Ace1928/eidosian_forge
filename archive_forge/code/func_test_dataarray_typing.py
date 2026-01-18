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
@pytest.mark.parametrize('label', data_type_codes.value_set('label'))
def test_dataarray_typing(label):
    dtype = data_type_codes.dtype[label]
    code = data_type_codes.code[label]
    arr = np.zeros((5,), dtype=dtype)
    if dtype in ('uint8', 'int32', 'float32'):
        assert GiftiDataArray(arr).datatype == code
    else:
        with pytest.raises(ValueError):
            GiftiDataArray(arr)
    assert GiftiDataArray(arr, datatype=label).datatype == code
    assert GiftiDataArray(arr, datatype=code).datatype == code
    if dtype != np.dtype('void'):
        assert GiftiDataArray(arr, datatype=dtype).datatype == code
    gda = GiftiDataArray()
    gda.data = arr
    gda.datatype = data_type_codes.code[label]
    assert gda.data.dtype == dtype
    assert gda.datatype == data_type_codes.code[label]