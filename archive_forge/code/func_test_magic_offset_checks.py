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
def test_magic_offset_checks(self):
    HC = self.header_class
    hdr = HC()
    hdr['magic'] = 'ooh'
    fhdr, message, raiser = self.log_chk(hdr, 45)
    assert fhdr['magic'] == b'ooh'
    assert message == "magic string 'ooh' is not valid; leaving as is, but future errors are likely"
    svo = hdr.single_vox_offset
    for magic, ok, bad_spm in ((hdr.pair_magic, 32, 40), (hdr.single_magic, svo + 32, svo + 40)):
        hdr['magic'] = magic
        hdr['vox_offset'] = 0
        self.assert_no_log_err(hdr)
        hdr['vox_offset'] = ok
        self.assert_no_log_err(hdr)
        hdr['vox_offset'] = bad_spm
        fhdr, message, raiser = self.log_chk(hdr, 30)
        assert fhdr['vox_offset'] == bad_spm
        assert message == f'vox offset (={bad_spm:g}) not divisible by 16, not SPM compatible; leaving at current value'
    hdr['magic'] = hdr.single_magic
    hdr['vox_offset'] = 10
    fhdr, message, raiser = self.log_chk(hdr, 40)
    assert fhdr['vox_offset'] == hdr.single_vox_offset
    assert message == 'vox offset 10 too low for single file nifti1; setting to minimum value of ' + str(hdr.single_vox_offset)