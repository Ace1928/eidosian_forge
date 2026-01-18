import os
import struct
import unittest
import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from nibabel import nifti1 as nifti1
from nibabel.affines import from_matvec
from nibabel.casting import have_binary128, type_info
from nibabel.eulerangles import euler2mat
from nibabel.nifti1 import (
from nibabel.optpkg import optional_package
from nibabel.pkg_info import cmp_pkg_version
from nibabel.spatialimages import HeaderDataError
from nibabel.tmpdirs import InTemporaryDirectory
from ..freesurfer import load as mghload
from ..orientations import aff2axcodes
from ..testing import (
from . import test_analyze as tana
from . import test_spm99analyze as tspm
from .nibabel_data import get_nibabel_data, needs_nibabel_data
from .test_arraywriters import IUINT_TYPES, rt_err_estimate
from .test_orientations import ALL_ORNTS
def test_load_pixdims(self):
    IC = self.image_class
    HC = IC.header_class
    arr = np.arange(24).reshape((2, 3, 4))
    qaff = np.diag([2, 3, 4, 1])
    saff = np.diag([5, 6, 7, 1])
    hdr = HC()
    hdr.set_qform(qaff)
    assert_array_equal(hdr.get_qform(), qaff)
    hdr.set_sform(saff)
    assert_array_equal(hdr.get_sform(), saff)
    simg = IC(arr, None, hdr)
    img_hdr = simg.header
    assert_array_equal(img_hdr.get_qform(), qaff)
    assert_array_equal(img_hdr.get_sform(), saff)
    assert_array_equal(img_hdr.get_zooms(), [2, 3, 4])
    re_simg = bytesio_round_trip(simg)
    assert_array_equal(re_simg.get_fdata(), arr)
    rimg_hdr = re_simg.header
    assert_array_equal(rimg_hdr.get_qform(), qaff)
    assert_array_equal(rimg_hdr.get_sform(), saff)
    assert_array_equal(rimg_hdr.get_zooms(), [2, 3, 4])