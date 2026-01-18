import itertools
import logging
import os
import pickle
import re
from io import BytesIO, StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import imageglobals
from ..analyze import AnalyzeHeader, AnalyzeImage
from ..arraywriters import WriterError
from ..casting import sctypes_aliases
from ..nifti1 import Nifti1Header
from ..optpkg import optional_package
from ..spatialimages import HeaderDataError, HeaderTypeError, supported_np_types
from ..testing import (
from ..tmpdirs import InTemporaryDirectory
from . import test_spatialimages as tsi
from . import test_wrapstruct as tws
def test_dtype_to_filename_arg(self):
    img_klass = self.image_class
    arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    aff = np.eye(4)
    img = img_klass(arr, aff)
    fname = 'test' + img_klass.files_types[0][1]
    with InTemporaryDirectory():
        for dtype in self.supported_np_types:
            img.to_filename(fname, dtype=dtype)
            new_img = img_klass.from_filename(fname)
            assert new_img.get_data_dtype() == dtype
            assert img.get_data_dtype() == np.int16