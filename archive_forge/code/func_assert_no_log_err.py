import logging
from io import BytesIO, StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import imageglobals
from ..batteryrunners import Report
from ..casting import sctypes
from ..spatialimages import HeaderDataError
from ..volumeutils import Recoder, native_code, swapped_code
from ..wrapstruct import LabeledWrapStruct, WrapStruct, WrapStructError
def assert_no_log_err(self, hdr):
    """Assert that no logging or errors result from this `hdr`"""
    fhdr, message, raiser = self.log_chk(hdr, 0)
    assert (fhdr, message) == (hdr, '')