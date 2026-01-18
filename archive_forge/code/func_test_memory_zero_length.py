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
def test_memory_zero_length(tmpdir):
    path = os.path.join(str(tmpdir), guid())
    f = open(path, 'wb')
    f.close()
    with pa.memory_map(path, mode='r+b') as memory_map:
        assert memory_map.size() == 0