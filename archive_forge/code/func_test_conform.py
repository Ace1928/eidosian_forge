import logging
from os.path import dirname
from os.path import join as pjoin
import numpy as np
import numpy.linalg as npl
from nibabel.optpkg import optional_package
import unittest
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
import nibabel as nib
from nibabel.affines import AffineError, apply_affine, from_matvec, to_matvec, voxel_sizes
from nibabel.eulerangles import euler2mat
from nibabel.nifti1 import Nifti1Image
from nibabel.nifti2 import Nifti2Image
from nibabel.orientations import aff2axcodes, inv_ornt_aff
from nibabel.processing import (
from nibabel.testing import assert_allclose_safely
from nibabel.tests.test_spaces import assert_all_in, get_outspace_params
from .test_imageclasses import MINC_3DS, MINC_4DS
@needs_scipy
def test_conform(caplog):
    anat = nib.load(pjoin(DATA_DIR, 'anatomical.nii'))
    c = conform(anat)
    assert c.shape == (256, 256, 256)
    assert c.header.get_zooms() == (1, 1, 1)
    assert c.dataobj.dtype.type == anat.dataobj.dtype.type
    assert aff2axcodes(c.affine) == ('R', 'A', 'S')
    assert isinstance(c, Nifti1Image)
    with caplog.at_level(logging.CRITICAL):
        c = conform(anat, out_shape=(100, 100, 200), voxel_size=(2, 2, 1.5), orientation='LPI', out_class=Nifti2Image)
    assert c.shape == (100, 100, 200)
    assert c.header.get_zooms() == (2, 2, 1.5)
    assert c.dataobj.dtype.type == anat.dataobj.dtype.type
    assert aff2axcodes(c.affine) == ('L', 'P', 'I')
    assert isinstance(c, Nifti2Image)
    func = nib.load(pjoin(DATA_DIR, 'functional.nii'))
    with pytest.raises(ValueError):
        conform(func)
    with pytest.raises(ValueError):
        conform(anat, out_shape=(100, 100))
    with pytest.raises(ValueError):
        conform(anat, voxel_size=(2, 2))