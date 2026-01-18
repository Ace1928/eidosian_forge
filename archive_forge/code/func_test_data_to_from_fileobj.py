import bz2
import gzip
import types
import warnings
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import Nifti1Image, load, minc1
from ..deprecated import ModuleProxy
from ..deprecator import ExpiredDeprecationError
from ..externals.netcdf import netcdf_file
from ..minc1 import Minc1File, Minc1Image, MincHeader
from ..optpkg import optional_package
from ..testing import assert_data_similar, clear_and_catch_warnings, data_path
from ..tmpdirs import InTemporaryDirectory
from . import test_spatialimages as tsi
from .test_fileslice import slicer_samples
def test_data_to_from_fileobj(self):
    for fpath in self.eg_images:
        img = self.module.load(fpath)
        bio = BytesIO()
        arr = np.arange(24).reshape((2, 3, 4))
        with pytest.raises(NotImplementedError):
            img.header.data_to_fileobj(arr, bio)
        with pytest.raises(NotImplementedError):
            img.header.data_from_fileobj(bio)