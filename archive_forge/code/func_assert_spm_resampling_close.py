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
def assert_spm_resampling_close(from_img, our_resampled, spm_resampled):
    """Assert our resampling is close to SPM's, allowing for edge effects"""
    to_img_shape = spm_resampled.shape
    to_img_affine = spm_resampled.affine
    to_vox_coords = np.indices(to_img_shape).transpose((1, 2, 3, 0))
    to_to_from = npl.inv(from_img.affine).dot(to_img_affine)
    resamp_coords = apply_affine(to_to_from, to_vox_coords)
    outside_vol = np.any((resamp_coords < 0) | (np.subtract(resamp_coords, from_img.shape) > -1), axis=-1)
    spm_res = np.where(outside_vol, np.nan, np.array(spm_resampled.dataobj))
    assert_allclose_safely(our_resampled.dataobj, spm_res)
    assert_almost_equal(our_resampled.affine, spm_resampled.affine, 5)