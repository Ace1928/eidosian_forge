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
def test_native_file_open_error():
    with assert_file_not_found():
        pa.OSFile('non_existent_file', 'rb')
    with assert_file_not_found():
        pa.memory_map('non_existent_file', 'rb')