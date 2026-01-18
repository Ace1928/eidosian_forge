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
def test_names_overwrite(self):
    descriptor = {'names': ('g', 'a', 'w'), 'formats': ('S1', 'i4', 'f4')}
    data = TextIO(b'M 64.0 75.0\nF 25.0 60.0')
    names = ('gender', 'age', 'weight')
    test = np.genfromtxt(data, dtype=descriptor, names=names)
    descriptor['names'] = names
    control = np.array([('M', 64.0, 75.0), ('F', 25.0, 60.0)], dtype=descriptor)
    assert_equal(test, control)