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
def test_inmemory_write_after_closed():
    f = pa.BufferOutputStream()
    f.write(b'ok')
    assert not f.closed
    f.getvalue()
    assert f.closed
    with pytest.raises(ValueError):
        f.write(b'not ok')