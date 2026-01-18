import bz2
from contextlib import contextmanager
from io import (BytesIO, StringIO, TextIOWrapper, BufferedIOBase, IOBase)
import itertools
import gc
import gzip
import math
import os
import pathlib
import pytest
import sys
import tempfile
import weakref
import numpy as np
from pyarrow.util import guid
from pyarrow import Codec
import pyarrow as pa
def test_native_file_permissions(tmpdir):
    cur_umask = os.umask(2)
    os.umask(cur_umask)
    path = os.path.join(str(tmpdir), guid())
    with pa.OSFile(path, mode='w'):
        pass
    assert os.stat(path).st_mode & 511 == 438 & ~cur_umask
    path = os.path.join(str(tmpdir), guid())
    with pa.memory_map(path, 'w'):
        pass
    assert os.stat(path).st_mode & 511 == 438 & ~cur_umask