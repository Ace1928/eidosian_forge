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
def test_fail_save():
    with InTemporaryDirectory():
        dataobj = np.ones((10, 10, 10), dtype=np.float16)
        affine = np.eye(4, dtype=np.float32)
        img = SpatialImage(dataobj, affine)
        with pytest.raises(AttributeError):
            nils.save(img, 'foo.nii.gz')
        del img