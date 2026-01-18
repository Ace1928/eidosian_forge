import itertools
import logging
import os
import pickle
import re
from io import BytesIO, StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import imageglobals
from ..analyze import AnalyzeHeader, AnalyzeImage
from ..arraywriters import WriterError
from ..casting import sctypes_aliases
from ..nifti1 import Nifti1Header
from ..optpkg import optional_package
from ..spatialimages import HeaderDataError, HeaderTypeError, supported_np_types
from ..testing import (
from ..tmpdirs import InTemporaryDirectory
from . import test_spatialimages as tsi
from . import test_wrapstruct as tws
def test_big_offset_exts(self):
    img_klass = self.image_class
    arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    aff = np.eye(4)
    img_ext = img_klass.files_types[0][1]
    compressed_exts = ['', '.gz', '.bz2']
    if HAVE_ZSTD:
        compressed_exts += ['.zst']
    with InTemporaryDirectory():
        for offset in (0, 2048):
            for compressed_ext in compressed_exts:
                img = img_klass(arr, aff)
                img.header.set_data_offset(offset)
                fname = 'test' + img_ext + compressed_ext
                img.to_filename(fname)
                img_back = img_klass.from_filename(fname)
                assert_array_equal(arr, img_back.dataobj)
        del img, img_back