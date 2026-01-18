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
def test_header(self):
    data = TextIO('gender age weight\nM 64.0 75.0\nF 25.0 60.0')
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
        test = np.genfromtxt(data, dtype=None, names=True)
        assert_(w[0].category is np.VisibleDeprecationWarning)
    control = {'gender': np.array([b'M', b'F']), 'age': np.array([64.0, 25.0]), 'weight': np.array([75.0, 60.0])}
    assert_equal(test['gender'], control['gender'])
    assert_equal(test['age'], control['age'])
    assert_equal(test['weight'], control['weight'])