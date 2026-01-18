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
@expires('5.0.0')
def test_guessed_image_type():
    with deprecated_to('5.0.0'):
        assert nils.guessed_image_type(pjoin(DATA_PATH, 'example4d.nii.gz')) == Nifti1Image
        assert nils.guessed_image_type(pjoin(DATA_PATH, 'nifti1.hdr')) == Nifti1Pair
        assert nils.guessed_image_type(pjoin(DATA_PATH, 'example_nifti2.nii.gz')) == Nifti2Image
        assert nils.guessed_image_type(pjoin(DATA_PATH, 'nifti2.hdr')) == Nifti2Pair
        assert nils.guessed_image_type(pjoin(DATA_PATH, 'tiny.mnc')) == Minc1Image
        assert nils.guessed_image_type(pjoin(DATA_PATH, 'small.mnc')) == Minc2Image
        assert nils.guessed_image_type(pjoin(DATA_PATH, 'test.mgz')) == MGHImage
        assert nils.guessed_image_type(pjoin(DATA_PATH, 'analyze.hdr')) == Spm2AnalyzeImage