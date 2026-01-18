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
def test_integer_delimiter(self):
    data = '  1  2  3\n  4  5 67\n890123  4'
    test = np.genfromtxt(TextIO(data), delimiter=3)
    control = np.array([[1, 2, 3], [4, 5, 67], [890, 123, 4]])
    assert_equal(test, control)