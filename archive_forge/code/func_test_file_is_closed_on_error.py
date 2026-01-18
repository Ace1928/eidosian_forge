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
def test_file_is_closed_on_error(self):
    with tempdir() as tmpdir:
        fpath = os.path.join(tmpdir, 'test.csv')
        with open(fpath, 'wb') as f:
            f.write('ϖ'.encode())
        with assert_no_warnings():
            with pytest.raises(UnicodeDecodeError):
                np.genfromtxt(fpath, encoding='ascii')