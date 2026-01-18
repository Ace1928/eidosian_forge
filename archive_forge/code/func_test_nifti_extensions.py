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
def test_nifti_extensions():
    nim = load(image_file)
    hdr = nim.header
    exts_container = hdr.extensions
    assert len(exts_container) == 2
    assert exts_container.count('comment') == 2
    assert exts_container.count('afni') == 0
    assert exts_container.get_codes() == [6, 6]
    assert exts_container.get_sizeondisk() % 16 == 0
    assert exts_container[0].get_content() == b'extcomment1'
    afniext = Nifti1Extension('afni', '<xml></xml>')
    exts_container.append(afniext)
    assert exts_container.get_codes() == [6, 6, 4]
    assert exts_container.count('comment') == 2
    assert exts_container.count('afni') == 1
    assert exts_container.get_sizeondisk() % 16 == 0
    del exts_container[1]
    assert exts_container.get_codes() == [6, 4]
    assert exts_container.count('comment') == 1
    assert exts_container.count('afni') == 1