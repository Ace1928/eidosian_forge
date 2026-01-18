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
def test_assert_parallel():
    dw = didw.wrapper_from_file(DATA_FILE_SLC_NORM)
    dw.image_orient_patient = np.c_[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    with pytest.raises(AssertionError):
        dw.slice_normal