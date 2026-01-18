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
def test_converters_with_usecols_and_names(self):
    data = TextIO('A B C D\n aaaa 121 45 9.1')
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
        test = np.genfromtxt(data, usecols=('A', 'C', 'D'), names=True, dtype=None, converters={'C': lambda s: 2 * int(s)})
        assert_(w[0].category is np.VisibleDeprecationWarning)
    control = np.array(('aaaa', 90, 9.1), dtype=[('A', '|S4'), ('C', int), ('D', float)])
    assert_equal(test, control)