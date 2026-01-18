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
def test_dtype_with_converters_and_usecols(self):
    dstr = '1,5,-1,1:1\n2,8,-1,1:n\n3,3,-2,m:n\n'
    dmap = {'1:1': 0, '1:n': 1, 'm:1': 2, 'm:n': 3}
    dtyp = [('e1', 'i4'), ('e2', 'i4'), ('e3', 'i2'), ('n', 'i1')]
    conv = {0: int, 1: int, 2: int, 3: lambda r: dmap[r.decode()]}
    test = np.recfromcsv(TextIO(dstr), dtype=dtyp, delimiter=',', names=None, converters=conv)
    control = np.rec.array([(1, 5, -1, 0), (2, 8, -1, 1), (3, 3, -2, 3)], dtype=dtyp)
    assert_equal(test, control)
    dtyp = [('e1', 'i4'), ('e2', 'i4'), ('n', 'i1')]
    test = np.recfromcsv(TextIO(dstr), dtype=dtyp, delimiter=',', usecols=(0, 1, 3), names=None, converters=conv)
    control = np.rec.array([(1, 5, 0), (2, 8, 1), (3, 3, 3)], dtype=dtyp)
    assert_equal(test, control)