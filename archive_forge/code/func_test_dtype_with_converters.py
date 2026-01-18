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
def test_dtype_with_converters(self):
    dstr = '2009; 23; 46'
    test = np.genfromtxt(TextIO(dstr), delimiter=';', dtype=float, converters={0: bytes})
    control = np.array([('2009', 23.0, 46)], dtype=[('f0', '|S4'), ('f1', float), ('f2', float)])
    assert_equal(test, control)
    test = np.genfromtxt(TextIO(dstr), delimiter=';', dtype=float, converters={0: float})
    control = np.array([2009.0, 23.0, 46])
    assert_equal(test, control)