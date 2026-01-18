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
def test_mincfile_slicing(self):
    for tp in self.test_files:
        mnc_obj = self.opener(tp['fname'], 'r')
        mnc = self.file_class(mnc_obj)
        data = mnc.get_scaled_data()
        for slicedef in ((slice(None),), (1,), (slice(None), 1), (1, slice(None)), (slice(None), 1, 1), (1, slice(None), 1), (1, 1, slice(None))):
            sliced_data = mnc.get_scaled_data(slicedef)
            assert_array_equal(sliced_data, data[slicedef])
        del mnc, data