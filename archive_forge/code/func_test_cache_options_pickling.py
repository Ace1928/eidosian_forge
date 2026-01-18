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
def test_cache_options_pickling(pickle_module):
    options = [pa.CacheOptions(), pa.CacheOptions(hole_size_limit=4096, range_size_limit=8192, lazy=True, prefetch_limit=5)]
    for option in options:
        assert pickle_module.loads(pickle_module.dumps(option)) == option