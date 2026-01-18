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
def test_max_rows(self):
    data = '1 2\n3 4\n5 6\n7 8\n9 10\n'
    txt = TextIO(data)
    a1 = np.genfromtxt(txt, max_rows=3)
    a2 = np.genfromtxt(txt)
    assert_equal(a1, [[1, 2], [3, 4], [5, 6]])
    assert_equal(a2, [[7, 8], [9, 10]])
    assert_raises(ValueError, np.genfromtxt, TextIO(data), max_rows=0)
    data = '1 1\n2 2\n0 \n3 3\n4 4\n5  \n6  \n7  \n'
    test = np.genfromtxt(TextIO(data), max_rows=2)
    control = np.array([[1.0, 1.0], [2.0, 2.0]])
    assert_equal(test, control)
    assert_raises(ValueError, np.genfromtxt, TextIO(data), skip_footer=1, max_rows=4)
    assert_raises(ValueError, np.genfromtxt, TextIO(data), max_rows=4)
    with suppress_warnings() as sup:
        sup.filter(ConversionWarning)
        test = np.genfromtxt(TextIO(data), max_rows=4, invalid_raise=False)
        control = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        assert_equal(test, control)
        test = np.genfromtxt(TextIO(data), max_rows=5, invalid_raise=False)
        control = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        assert_equal(test, control)
    data = 'a b\n#c d\n1 1\n2 2\n#0 \n3 3\n4 4\n5  5\n'
    txt = TextIO(data)
    test = np.genfromtxt(txt, skip_header=1, max_rows=3, names=True)
    control = np.array([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], dtype=[('c', '<f8'), ('d', '<f8')])
    assert_equal(test, control)
    test = np.genfromtxt(txt, max_rows=None, dtype=test.dtype)
    control = np.array([(4.0, 4.0), (5.0, 5.0)], dtype=[('c', '<f8'), ('d', '<f8')])
    assert_equal(test, control)