import logging
import pathlib
import shutil
from io import BytesIO
from os.path import dirname
from os.path import join as pjoin
from tempfile import mkdtemp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import (
from .. import analyze as ana
from .. import loadsave as nils
from .. import nifti1 as ni1
from .. import spm2analyze as spm2
from .. import spm99analyze as spm99
from ..optpkg import optional_package
from ..spatialimages import SpatialImage
from ..testing import deprecated_to, expires
from ..tmpdirs import InTemporaryDirectory
from ..volumeutils import native_code, swapped_code
def test_filename_save():
    inklass_ext_loadklasses = ((Nifti1Image, '.nii', Nifti1Image), (Nifti2Image, '.nii', Nifti2Image), (Nifti1Pair, '.nii', Nifti1Image), (Nifti2Pair, '.nii', Nifti2Image), (Nifti1Image, '.img', Nifti1Pair), (Nifti2Image, '.img', Nifti2Pair), (Nifti1Pair, '.img', Nifti1Pair), (Nifti2Pair, '.img', Nifti2Pair), (Nifti1Image, '.hdr', Nifti1Pair), (Nifti2Image, '.hdr', Nifti2Pair), (Nifti1Pair, '.hdr', Nifti1Pair), (Nifti2Pair, '.hdr', Nifti2Pair), (Minc1Image, '.nii', Nifti1Image), (Minc1Image, '.img', Nifti1Pair), (Spm2AnalyzeImage, '.nii', Nifti1Image), (Spm2AnalyzeImage, '.img', Spm2AnalyzeImage), (Spm99AnalyzeImage, '.nii', Nifti1Image), (Spm99AnalyzeImage, '.img', Spm2AnalyzeImage), (AnalyzeImage, '.nii', Nifti1Image), (AnalyzeImage, '.img', Spm2AnalyzeImage))
    shape = (2, 4, 6)
    affine = np.diag([1, 2, 3, 1])
    data = np.arange(np.prod(shape), dtype='f4').reshape(shape)
    for inklass, out_ext, loadklass in inklass_ext_loadklasses:
        if not have_scipy:
            if ('mat', '.mat') in loadklass.files_types:
                continue
        img = inklass(data, affine)
        try:
            pth = mkdtemp()
            fname = pjoin(pth, 'image' + out_ext)
            for path in (fname, pathlib.Path(fname)):
                nils.save(img, path)
                rt_img = nils.load(path)
                assert_array_almost_equal(rt_img.get_fdata(), data)
                assert type(rt_img) is loadklass
                del rt_img
        finally:
            shutil.rmtree(pth)