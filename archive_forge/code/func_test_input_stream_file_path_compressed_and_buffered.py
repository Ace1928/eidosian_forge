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
def test_input_stream_file_path_compressed_and_buffered(tmpdir):
    data = b'some test data\n' * 100 + b'eof\n'
    gz_data = gzip.compress(data)
    file_path = tmpdir / 'input_stream_compressed_and_buffered.gz'
    with open(str(file_path), 'wb') as f:
        f.write(gz_data)
    stream = pa.input_stream(file_path, buffer_size=32, compression='gzip')
    assert stream.read() == data
    stream = pa.input_stream(str(file_path), buffer_size=64)
    assert stream.read() == data
    stream = pa.input_stream(pathlib.Path(str(file_path)), buffer_size=1024)
    assert stream.read() == data