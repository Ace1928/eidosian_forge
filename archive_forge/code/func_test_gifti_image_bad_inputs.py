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
def test_gifti_image_bad_inputs():
    img = GiftiImage()
    pytest.raises(TypeError, img.add_gifti_data_array, 'not-a-data-array')

    def assign_labeltable(val):
        img.labeltable = val
    pytest.raises(TypeError, assign_labeltable, 'not-a-table')

    def assign_metadata(val):
        img.meta = val
    pytest.raises(TypeError, assign_metadata, 'not-a-meta')