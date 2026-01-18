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
def test_comments_multi_chars(self):
    c = TextIO()
    c.write('/* comment\n1,2,3,5\n')
    c.seek(0)
    x = np.loadtxt(c, dtype=int, delimiter=',', comments='/*')
    a = np.array([1, 2, 3, 5], int)
    assert_array_equal(x, a)
    c = TextIO()
    c.write('*/ comment\n1,2,3,5\n')
    c.seek(0)
    assert_raises(ValueError, np.loadtxt, c, dtype=int, delimiter=',', comments='/*')