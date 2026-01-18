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
def test_invalid_converter(self):
    strip_rand = lambda x: float(b'r' in x.lower() and x.split()[-1] or (b'r' not in x.lower() and x.strip() or 0.0))
    strip_per = lambda x: float(b'%' in x.lower() and x.split()[0] or (b'%' not in x.lower() and x.strip() or 0.0))
    s = TextIO('D01N01,10/1/2003 ,1 %,R 75,400,600\r\nL24U05,12/5/2003, 2 %,1,300, 150.5\r\nD02N03,10/10/2004,R 1,,7,145.55')
    kwargs = dict(converters={2: strip_per, 3: strip_rand}, delimiter=',', dtype=None)
    assert_raises(ConverterError, np.genfromtxt, s, **kwargs)