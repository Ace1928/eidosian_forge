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
@pytest.mark.xfail(reason='Not packaged in install', raises=FileNotFoundError)
def test_data_real(self):
    dw = didw.wrapper_from_file(DATA_FILE_4D)
    data = dw.get_data()
    if endian_codes[data.dtype.byteorder] == '>':
        data = data.byteswap()
    dat_str = data.tobytes()
    assert sha1(dat_str).hexdigest() == '149323269b0af92baa7508e19ca315240f77fa8c'