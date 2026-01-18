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
def test_input_stream_errors(tmpdir):
    buf = memoryview(b'')
    with pytest.raises(ValueError):
        pa.input_stream(buf, compression='foo')
    for arg in [bytearray(), StringIO()]:
        with pytest.raises(TypeError):
            pa.input_stream(arg)
    with assert_file_not_found():
        pa.input_stream('non_existent_file')
    with open(str(tmpdir / 'new_file'), 'wb') as f:
        with pytest.raises(TypeError, match='readable file expected'):
            pa.input_stream(f)