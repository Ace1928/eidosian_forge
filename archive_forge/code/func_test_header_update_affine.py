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
def test_header_update_affine(self):
    img = self.image_class(np.zeros((2, 3, 4)), np.eye(4))
    hdr = img.header
    aff = img.affine
    aff[:] = np.diag([1.1, 1.1, 1.1, 1])
    hdr.set_qform(aff, 2)
    hdr.set_sform(aff, 2)
    img.update_header()
    assert hdr['sform_code'] == 2
    assert hdr['qform_code'] == 2