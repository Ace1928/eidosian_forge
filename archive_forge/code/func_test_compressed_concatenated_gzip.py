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
@pytest.mark.gzip
def test_compressed_concatenated_gzip(tmpdir):
    data = b'some test data\n' * 10 + b'eof\n'
    fn = str(tmpdir / 'compressed_input_test2.gz')
    with gzip.open(fn, 'wb') as f:
        f.write(data[:50])
    with gzip.open(fn, 'ab') as f:
        f.write(data[50:])
    check_compressed_concatenated(data, fn, 'gzip')