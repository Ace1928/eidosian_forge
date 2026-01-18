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
def test_named_arrays(self):
    a = np.array([[1, 2], [3, 4]], float)
    b = np.array([[1 + 2j, 2 + 7j], [3 - 6j, 4 + 12j]], complex)
    c = BytesIO()
    np.savez(c, file_a=a, file_b=b)
    c.seek(0)
    l = np.load(c)
    assert_equal(a, l['file_a'])
    assert_equal(b, l['file_b'])