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
def test_affines_init(self):
    IC = self.image_class
    arr = np.arange(24, dtype='f4').reshape((2, 3, 4))
    aff = np.diag([2, 3, 4, 1])
    img = IC(arr, aff)
    hdr = img.header
    assert hdr['qform_code'] == 0
    assert hdr['sform_code'] == 2
    assert_array_equal(hdr.get_zooms(), [2, 3, 4])
    qaff = np.diag([3, 4, 5, 1])
    saff = np.diag([6, 7, 8, 1])
    hdr.set_qform(qaff, code='scanner')
    hdr.set_sform(saff, code='talairach')
    assert_array_equal(hdr.get_zooms(), [3, 4, 5])
    img = IC(arr, aff, hdr)
    new_hdr = img.header
    assert new_hdr['qform_code'] == 0
    assert new_hdr['sform_code'] == 2
    assert_array_equal(new_hdr.get_sform(), aff)
    assert_array_equal(new_hdr.get_zooms(), [2, 3, 4])
    img = IC(arr, None, hdr)
    new_hdr = img.header
    assert new_hdr['qform_code'] == 1
    assert_array_equal(new_hdr.get_qform(), qaff)
    assert new_hdr['sform_code'] == 3
    assert_array_equal(new_hdr.get_sform(), saff)
    assert_array_equal(new_hdr.get_zooms(), [3, 4, 5])