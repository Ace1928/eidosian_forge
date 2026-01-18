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
def test_memory_map_reader(sample_disk_data):
    _check_native_file_reader(pa.memory_map, sample_disk_data, allow_read_out_of_bounds=False)