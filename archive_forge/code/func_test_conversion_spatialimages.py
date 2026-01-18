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
def test_conversion_spatialimages(caplog):
    shape = (2, 4, 6)
    affine = np.diag([1, 2, 3, 1])
    klasses = [klass for klass in all_image_classes if klass.rw and issubclass(klass, SpatialImage)]
    for npt in (np.float32, np.int16):
        data = np.arange(np.prod(shape), dtype=npt).reshape(shape)
        for r_class in klasses:
            if not r_class.makeable:
                continue
            img = r_class(data, affine)
            img.set_data_dtype(npt)
            for w_class in klasses:
                if not w_class.makeable:
                    continue
                with caplog.at_level(logging.CRITICAL):
                    img2 = w_class.from_image(img)
                assert_array_equal(img2.get_fdata(), data)
                assert_array_equal(img2.affine, affine)