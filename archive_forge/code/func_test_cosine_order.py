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
def test_cosine_order():
    data = np.arange(60, dtype=np.int32).reshape((3, 4, 5))
    aff = np.diag([2.0, 3, 4, 1])
    aff[0] = [2, 1, 0, 10]
    img = MGHImage(data, aff)
    assert_almost_equal(img.affine, aff, 6)
    img_fobj = io.BytesIO()
    img2 = _mgh_rt(img, img_fobj)
    hdr2 = img2.header
    RZS = aff[:3, :3]
    zooms = np.sqrt(np.sum(RZS ** 2, axis=0))
    assert_almost_equal(hdr2['Mdc'].T, RZS / zooms)
    assert_almost_equal(hdr2['delta'], zooms)