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
def test_names_with_usecols_bug1636(self):
    data = 'A,B,C,D,E\n0,1,2,3,4\n0,1,2,3,4\n0,1,2,3,4'
    ctrl_names = ('A', 'C', 'E')
    test = np.genfromtxt(TextIO(data), dtype=(int, int, int), delimiter=',', usecols=(0, 2, 4), names=True)
    assert_equal(test.dtype.names, ctrl_names)
    test = np.genfromtxt(TextIO(data), dtype=(int, int, int), delimiter=',', usecols=('A', 'C', 'E'), names=True)
    assert_equal(test.dtype.names, ctrl_names)
    test = np.genfromtxt(TextIO(data), dtype=int, delimiter=',', usecols=('A', 'C', 'E'), names=True)
    assert_equal(test.dtype.names, ctrl_names)