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
def test_nifti_xform_codes(self):
    hdr = self.header_class()
    affine = np.eye(4)
    for code in nifti1.xform_codes.keys():
        hdr.set_qform(affine, code)
        assert hdr['qform_code'] == nifti1.xform_codes[code]
        hdr.set_sform(affine, code)
        assert hdr['sform_code'] == nifti1.xform_codes[code]
    for bad_code in (-1, 6, 10):
        with pytest.raises(KeyError):
            hdr.set_qform(affine, bad_code)
        with pytest.raises(KeyError):
            hdr.set_sform(affine, bad_code)