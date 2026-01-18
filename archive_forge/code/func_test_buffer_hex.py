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
@pytest.mark.parametrize('val, expected_hex_buffer', [(b'check', b'636865636B'), (b'\x070', b'0730'), (b'', b'')])
def test_buffer_hex(val, expected_hex_buffer):
    buf = pa.py_buffer(val)
    assert buf.hex() == expected_hex_buffer