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
def check_compressed(self, fopen, suffixes):
    wanted = np.arange(6).reshape((2, 3))
    linesep = ('\n', '\r\n', '\r')
    for sep in linesep:
        data = '0 1 2' + sep + '3 4 5'
        for suffix in suffixes:
            with temppath(suffix=suffix) as name:
                with fopen(name, mode='wt', encoding='UTF-32-LE') as f:
                    f.write(data)
                res = self.loadfunc(name, encoding='UTF-32-LE')
                assert_array_equal(res, wanted)
                with fopen(name, 'rt', encoding='UTF-32-LE') as f:
                    res = self.loadfunc(f)
                assert_array_equal(res, wanted)