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
def check_buffer_pickling(buf, pickler):
    for protocol in range(0, pickler.HIGHEST_PROTOCOL + 1):
        result = pickler.loads(pickler.dumps(buf, protocol=protocol))
        assert len(result) == len(buf)
        assert memoryview(result) == memoryview(buf)
        assert result.to_pybytes() == buf.to_pybytes()
        assert result.is_mutable == buf.is_mutable