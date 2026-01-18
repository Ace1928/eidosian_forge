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
def test_binary_decode_autodtype(self):
    utf16 = b'\xff\xfeh\x04 \x00i\x04 \x00j\x04'
    v = self.loadfunc(BytesIO(utf16), dtype=None, encoding='UTF-16')
    assert_array_equal(v, np.array(utf16.decode('UTF-16').split()))