import shutil
import sys
import warnings
from os.path import basename, dirname
from os.path import join as pjoin
from unittest import mock
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from ...loadsave import load, save
from ...nifti1 import xform_codes
from ...testing import clear_and_catch_warnings, suppress_warnings
from ...tmpdirs import InTemporaryDirectory
from .. import gifti as gi
from ..parse_gifti_fast import GiftiImageParser, GiftiParseError
from ..util import gifti_endian_codes
def test_external_file_failure_cases():
    with InTemporaryDirectory() as tmpdir:
        shutil.copy(DATA_FILE7, '.')
        filename = pjoin(tmpdir, basename(DATA_FILE7))
        with pytest.raises(GiftiParseError):
            img = load(filename)
    with open(DATA_FILE7, 'rb') as f:
        xmldata = f.read()
    parser = GiftiImageParser()
    with pytest.raises(GiftiParseError):
        img = parser.parse(xmldata)