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
def check_roundtrips(self, a):
    self.roundtrip(a)
    self.roundtrip(a, file_on_disk=True)
    self.roundtrip(np.asfortranarray(a))
    self.roundtrip(np.asfortranarray(a), file_on_disk=True)
    if a.shape[0] > 1:
        self.roundtrip(np.asfortranarray(a)[1:])
        self.roundtrip(np.asfortranarray(a)[1:], file_on_disk=True)