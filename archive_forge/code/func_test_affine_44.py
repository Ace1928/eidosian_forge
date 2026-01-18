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
def test_affine_44(self):
    IC = self.image_class
    shape = (2, 3, 4)
    data = np.arange(24, dtype=np.int16).reshape(shape)
    affine = np.diag([2, 3, 4, 1])
    img = IC(data, affine)
    assert_array_equal(affine, img.affine)
    img = IC(data, affine.tolist())
    assert_array_equal(affine, img.affine)
    with pytest.raises(ValueError):
        IC(data, np.diag([2, 3, 4]))