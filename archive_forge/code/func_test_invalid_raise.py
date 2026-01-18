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
def test_invalid_raise(self):
    data = ['1, 1, 1, 1, 1'] * 50
    for i in range(5):
        data[10 * i] = '2, 2, 2, 2 2'
    data.insert(0, 'a, b, c, d, e')
    mdata = TextIO('\n'.join(data))
    kwargs = dict(delimiter=',', dtype=None, names=True)

    def f():
        return np.genfromtxt(mdata, invalid_raise=False, **kwargs)
    mtest = assert_warns(ConversionWarning, f)
    assert_equal(len(mtest), 45)
    assert_equal(mtest, np.ones(45, dtype=[(_, int) for _ in 'abcde']))
    mdata.seek(0)
    assert_raises(ValueError, np.genfromtxt, mdata, delimiter=',', names=True)