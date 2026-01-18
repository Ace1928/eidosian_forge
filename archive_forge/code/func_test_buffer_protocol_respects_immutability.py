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
def test_buffer_protocol_respects_immutability():
    a = b'12345'
    arrow_ref = pa.py_buffer(a)
    numpy_ref = np.frombuffer(arrow_ref, dtype=np.uint8)
    assert not numpy_ref.flags.writeable