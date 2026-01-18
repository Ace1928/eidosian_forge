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
@pytest.mark.slow
def test_format_2_0(self):
    dt = [('%d' % i * 100, float) for i in range(500)]
    a = np.ones(1000, dtype=dt)
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings('always', '', UserWarning)
        self.check_roundtrips(a)