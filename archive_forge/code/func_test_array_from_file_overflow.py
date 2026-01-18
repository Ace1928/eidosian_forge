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
def test_array_from_file_overflow():
    shape = (1500,) * 6

    class NoStringIO:

        def seek(self, n_bytes):
            pass

        def read(self, n_bytes):
            return b''
    try:
        array_from_file(shape, np.int8, NoStringIO())
    except OSError as err:
        message = str(err)
    assert message == 'Expected 11390625000000000000 bytes, got 0 bytes from object\n - could the file be damaged?'