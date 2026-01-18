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
def test_log_checks(self):
    HC = self.header_class
    hdr = HC()
    with suppress_warnings():
        hdr['sizeof_hdr'] = 350
        fhdr, message, raiser = self.log_chk(hdr, 30)
    assert fhdr['sizeof_hdr'] == self.sizeof_hdr
    assert message == f'sizeof_hdr should be {self.sizeof_hdr}; set sizeof_hdr to {self.sizeof_hdr}'
    pytest.raises(*raiser)
    hdr = HC()
    hdr.set_data_dtype('RGB')
    fhdr, message, raiser = self.log_chk(hdr, 0)
    hdr = HC()
    hdr['datatype'] = -1
    with suppress_warnings():
        fhdr, message, raiser = self.log_chk(hdr, 40)
    assert message == 'data code -1 not recognized; not attempting fix'
    pytest.raises(*raiser)
    hdr['datatype'] = 255
    fhdr, message, raiser = self.log_chk(hdr, 40)
    assert message == 'data code 255 not supported; not attempting fix'
    pytest.raises(*raiser)
    hdr = HC()
    hdr['datatype'] = 16
    hdr['bitpix'] = 16
    fhdr, message, raiser = self.log_chk(hdr, 10)
    assert fhdr['bitpix'] == 32
    assert message == 'bitpix does not match datatype; setting bitpix to match datatype'
    pytest.raises(*raiser)