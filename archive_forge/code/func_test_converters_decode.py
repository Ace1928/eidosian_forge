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
def test_converters_decode(self):
    c = TextIO()
    c.write(b'\xcf\x96')
    c.seek(0)
    x = self.loadfunc(c, dtype=np.str_, converters={0: lambda x: x.decode('UTF-8')})
    a = np.array([b'\xcf\x96'.decode('UTF-8')])
    assert_array_equal(x, a)