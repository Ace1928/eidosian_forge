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
@pytest.mark.skipif(locale.getpreferredencoding() == 'ANSI_X3.4-1968', reason='Wrong preferred encoding')
def test_binary_load(self):
    butf8 = b'5,6,7,\xc3\x95scarscar\r\n15,2,3,hello\r\n20,2,3,\xc3\x95scar\r\n'
    sutf8 = butf8.decode('UTF-8').replace('\r', '').splitlines()
    with temppath() as path:
        with open(path, 'wb') as f:
            f.write(butf8)
        with open(path, 'rb') as f:
            x = np.loadtxt(f, encoding='UTF-8', dtype=np.str_)
        assert_array_equal(x, sutf8)
        with open(path, 'rb') as f:
            x = np.loadtxt(f, encoding='UTF-8', dtype='S')
        x = [b'5,6,7,\xc3\x95scarscar', b'15,2,3,hello', b'20,2,3,\xc3\x95scar']
        assert_array_equal(x, np.array(x, dtype='S'))