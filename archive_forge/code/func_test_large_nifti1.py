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
@runif_extra_has('slow')
def test_large_nifti1():
    image_shape = (91, 109, 91, 1200)
    img = Nifti1Image(np.ones(image_shape, dtype=np.float32), affine=np.eye(4))
    with InTemporaryDirectory():
        img.to_filename('test.nii.gz')
        del img
        data = load('test.nii.gz').get_fdata()
    assert image_shape == data.shape
    n_ones = np.sum(data == 1.0)
    assert np.prod(image_shape) == n_ones