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
def test_dataarray_empty():
    null_da = GiftiDataArray()
    assert null_da.data is None
    assert null_da.intent == 0
    assert null_da.datatype == 0
    assert null_da.encoding == 3
    assert null_da.endian == (2 if sys.byteorder == 'little' else 1)
    assert null_da.coordsys.dataspace == 0
    assert null_da.coordsys.xformspace == 0
    assert_array_equal(null_da.coordsys.xform, np.eye(4))
    assert null_da.ind_ord == 1
    assert null_da.meta == {}
    assert null_da.ext_fname == ''
    assert null_da.ext_offset == 0