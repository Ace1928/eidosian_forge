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
def buf_chk(in_arr, out_buf, in_buf, offset):
    """Write contents of in_arr into fileobj, read back, check same"""
    instr = b' ' * offset + in_arr.tobytes(order='F')
    out_buf.write(instr)
    out_buf.flush()
    if in_buf is None:
        out_buf.seek(0)
        in_buf = out_buf
    arr = array_from_file(in_arr.shape, in_arr.dtype, in_buf, offset)
    return np.allclose(in_arr, arr)