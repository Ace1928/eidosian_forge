import sys
import gc
import gzip
import os
import threading
import time
import warnings
import io
import re
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile
from io import BytesIO, StringIO
from datetime import datetime
import locale
from multiprocessing import Value, get_context
from ctypes import c_bool
import numpy as np
import numpy.ma as ma
from numpy.lib._iotools import ConverterError, ConversionWarning
from numpy.compat import asbytes
from numpy.ma.testutils import assert_equal
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8), reason='PyPy bug in error formatting')
def test_default_float_converter_no_default_hex_conversion(self):
    """
        Ensure that fromhex is only used for values with the correct prefix and
        is not called by default. Regression test related to gh-19598.
        """
    c = TextIO('a b c')
    with pytest.raises(ValueError, match=".*convert string 'a' to float64 at row 0, column 1"):
        np.loadtxt(c)