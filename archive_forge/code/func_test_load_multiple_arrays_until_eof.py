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
def test_load_multiple_arrays_until_eof():
    f = BytesIO()
    np.save(f, 1)
    np.save(f, 2)
    f.seek(0)
    assert np.load(f) == 1
    assert np.load(f) == 2
    with pytest.raises(EOFError):
        np.load(f)