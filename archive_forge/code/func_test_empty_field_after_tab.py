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
def test_empty_field_after_tab(self):
    c = TextIO()
    c.write('1 \t2 \t3\tstart \n4\t5\t6\t  \n7\t8\t9.5\t')
    c.seek(0)
    dt = {'names': ('x', 'y', 'z', 'comment'), 'formats': ('<i4', '<i4', '<f4', '|S8')}
    x = np.loadtxt(c, dtype=dt, delimiter='\t')
    a = np.array([b'start ', b'  ', b''])
    assert_array_equal(x['comment'], a)