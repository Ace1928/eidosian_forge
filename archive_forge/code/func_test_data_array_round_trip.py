import itertools
import sys
import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nibabel.tmpdirs import InTemporaryDirectory
from ... import load
from ...fileholders import FileHolder
from ...nifti1 import data_type_codes
from ...testing import get_test_data
from .. import (
from .test_parse_gifti_fast import (
def test_data_array_round_trip():
    verts = np.zeros((4, 3), np.float32)
    verts[0, 0] = 10.5
    verts[1, 1] = 20.5
    verts[2, 2] = 30.5
    vertices = GiftiDataArray(verts)
    img = GiftiImage()
    img.add_gifti_data_array(vertices)
    bio = BytesIO()
    fmap = dict(image=FileHolder(fileobj=bio))
    bio.write(img.to_xml())
    bio.seek(0)
    gio = GiftiImage.from_file_map(fmap)
    vertices = gio.darrays[0].data
    assert_array_equal(vertices, verts)