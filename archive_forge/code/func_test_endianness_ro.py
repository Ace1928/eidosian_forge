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
def test_endianness_ro(self):
    """Its use in initialization tested in the init tests.
        Endianness gives endian interpretation of binary data. It is
        read only because the only common use case is to set the
        endianness on initialization (or occasionally byteswapping the
        data) - but this is done via via the as_byteswapped method
        """
    hdr = self.header_class()
    with pytest.raises(AttributeError):
        hdr.endianness = '<'