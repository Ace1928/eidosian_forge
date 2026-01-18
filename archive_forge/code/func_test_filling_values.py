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
def test_filling_values(self):
    data = b'1, 2, 3\n1, , 5\n0, 6, \n'
    kwargs = dict(delimiter=',', dtype=None, filling_values=-999)
    ctrl = np.array([[1, 2, 3], [1, -999, 5], [0, 6, -999]], dtype=int)
    test = np.genfromtxt(TextIO(data), **kwargs)
    assert_equal(test, ctrl)