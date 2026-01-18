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
def test_converters_nodecode(self):
    utf8 = b'\xcf\x96'.decode('UTF-8')
    with temppath() as path:
        with io.open(path, 'wt', encoding='UTF-8') as f:
            f.write(utf8)
        x = self.loadfunc(path, dtype=np.str_, converters={0: lambda x: x + 't'}, encoding='UTF-8')
        a = np.array([utf8 + 't'])
        assert_array_equal(x, a)