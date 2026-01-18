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
def test_no_finite_values(self):
    data = np.zeros((2, 3, 4))
    data[:, 0] = np.nan
    data[:, 1] = np.inf
    data[:, 2] = -np.inf
    img = self.image_class(data, None)
    img.set_data_dtype(np.int16)
    assert img.get_data_dtype() == np.dtype(np.int16)
    fm = bytesio_filemap(img)
    if not img.header.has_data_slope:
        with pytest.raises(WriterError):
            img.to_file_map(fm)
        return
    img.to_file_map(fm)
    img_back = self.image_class.from_file_map(fm)
    assert_array_equal(img_back.dataobj, 0)