import bz2
import functools
import gzip
import itertools
import os
import tempfile
import threading
import time
import warnings
from io import BytesIO
from os.path import exists
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from packaging.version import Version
from nibabel.testing import (
from ..casting import OK_FLOATS, floor_log2, sctypes, shared_range, type_info
from ..openers import BZ2File, ImageOpener, Opener
from ..optpkg import optional_package
from ..tmpdirs import InTemporaryDirectory
from ..volumeutils import (
def test__ftype4scaled_finite_warningfilters():
    finfo = np.finfo(np.float32)
    tst_arr = np.array((finfo.min, finfo.max), dtype=np.float32)
    go = threading.Event()
    stop = threading.Event()
    err = []

    class MakeTotalDestroy(threading.Thread):

        def run(self):
            with warnings.catch_warnings():
                go.set()
                while not stop.is_set():
                    warnings.filters[:] = []
                    time.sleep(0)

    class CheckScaling(threading.Thread):

        def run(self):
            go.wait()
            for i in range(200):
                try:
                    _ftype4scaled_finite(tst_arr, 2.0, 1.0, default=np.float16)
                except Exception as e:
                    err.append(e)
                    break
            stop.set()
    thread_a = CheckScaling()
    thread_b = MakeTotalDestroy()
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()
    if err:
        raise err[0]