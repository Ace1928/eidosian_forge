import getpass
import hashlib
import os
import struct
import time
import unittest
from os.path import isdir
from os.path import join as pjoin
from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose
from ...fileslice import strided_scalar
from ...testing import clear_and_catch_warnings
from ...tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from ...tmpdirs import InTemporaryDirectory
from .. import (
from ..io import _pack_rgb
@freesurfer_test
def test_morph_data():
    """Test IO of morphometry data file (eg. curvature)."""
    curv_path = pjoin(data_path, 'surf', 'lh.curv')
    curv = read_morph_data(curv_path)
    assert -1.0 < curv.min() < 0
    assert 0 < curv.max() < 1.0
    with InTemporaryDirectory():
        new_path = 'test'
        write_morph_data(new_path, curv)
        curv2 = read_morph_data(new_path)
        assert np.array_equal(curv2, curv)