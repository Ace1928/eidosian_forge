import pathlib
import shutil
from os.path import dirname
from os.path import join as pjoin
from tempfile import TemporaryDirectory
import numpy as np
from .. import (
from ..filebasedimages import ImageFileError
from ..loadsave import _signature_matches_extension, load, read_img_data
from ..openers import Opener
from ..optpkg import optional_package
from ..testing import deprecated_to, expires
from ..tmpdirs import InTemporaryDirectory
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
def test_load_empty_image():
    with InTemporaryDirectory():
        open('empty.nii', 'w').close()
        with pytest.raises(ImageFileError) as err:
            load('empty.nii')
    assert str(err.value).startswith('Empty file: ')