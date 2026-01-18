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
def test_dtype_with_object_no_converter(self):
    parsed = np.genfromtxt(TextIO('1'), dtype=object)
    assert parsed[()] == b'1'
    parsed = np.genfromtxt(TextIO('string'), dtype=object)
    assert parsed[()] == b'string'