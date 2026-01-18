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
def test_bad_usecols(self):
    with pytest.raises(OverflowError):
        np.loadtxt(['1\n'], usecols=[2 ** 64], delimiter=',')
    with pytest.raises((ValueError, OverflowError)):
        np.loadtxt(['1\n'], usecols=[2 ** 62], delimiter=',')
    with pytest.raises(TypeError, match='If a structured dtype .*. But 1 usecols were given and the number of fields is 3.'):
        np.loadtxt(['1,1\n'], dtype='i,(2)i', usecols=[0], delimiter=',')