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
def test_against_spm_resample():
    anat = nib.load(pjoin(DATA_DIR, 'anatomical.nii'))
    func = nib.load(pjoin(DATA_DIR, 'functional.nii'))
    some_rotations = euler2mat(0.1, 0.2, 0.3)
    extra_affine = from_matvec(some_rotations, [3, 4, 5])
    moved_anat = nib.Nifti1Image(anat.get_fdata(), extra_affine.dot(anat.affine), anat.header)
    one_func = nib.Nifti1Image(func.dataobj[..., 0], func.affine, func.header)
    moved2func = resample_from_to(moved_anat, one_func, order=1, cval=np.nan)
    spm_moved = nib.load(pjoin(DATA_DIR, 'resampled_anat_moved.nii'))
    assert_spm_resampling_close(moved_anat, moved2func, spm_moved)
    moved2output = resample_to_output(moved_anat, 4, order=1, cval=np.nan)
    spm2output = nib.load(pjoin(DATA_DIR, 'reoriented_anat_moved.nii'))
    assert_spm_resampling_close(moved_anat, moved2output, spm2output)