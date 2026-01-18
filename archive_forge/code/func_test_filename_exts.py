import io
import os
import pathlib
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from ... import imageglobals
from ...fileholders import FileHolder
from ...openers import ImageOpener
from ...spatialimages import HeaderDataError
from ...testing import data_path
from ...tests import test_spatialimages as tsi
from ...tests import test_wrapstruct as tws
from ...tmpdirs import InTemporaryDirectory
from ...volumeutils import sys_is_le
from ...wrapstruct import WrapStructError
from .. import load, save
from ..mghformat import MGHError, MGHHeader, MGHImage
def test_filename_exts():
    v = np.ones((7, 13, 3, 22), np.uint8)
    img = MGHImage(v, None)
    for ext in ('.mgh', '.mgz'):
        with InTemporaryDirectory():
            fname = 'tmpname' + ext
            save(img, fname)
            img_back = load(fname)
            assert_array_equal(img_back.get_fdata(), v)
            del img_back