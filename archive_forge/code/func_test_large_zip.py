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
@pytest.mark.skipif(sys.platform == 'win32', reason='files>4GB may not work')
@pytest.mark.slow
@requires_memory(free_bytes=7000000000.0)
def test_large_zip(self):

    def check_large_zip(memoryerror_raised):
        memoryerror_raised.value = False
        try:
            test_data = np.asarray([np.random.rand(np.random.randint(50, 100), 4) for i in range(800000)], dtype=object)
            with tempdir() as tmpdir:
                np.savez(os.path.join(tmpdir, 'test.npz'), test_data=test_data)
        except MemoryError:
            memoryerror_raised.value = True
            raise
    memoryerror_raised = Value(c_bool)
    ctx = get_context('fork')
    p = ctx.Process(target=check_large_zip, args=(memoryerror_raised,))
    p.start()
    p.join()
    if memoryerror_raised.value:
        raise MemoryError('Child process raised a MemoryError exception')
    if p.exitcode == -9:
        pytest.xfail('subprocess got a SIGKILL, apparently free memory was not sufficient')
    assert p.exitcode == 0