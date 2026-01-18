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
def test_input_stream_duck_typing():

    class DuckReader:

        def close(self):
            pass

        @property
        def closed(self):
            return False

        def read(self, nbytes=None):
            return b'hello'
    stream = pa.input_stream(DuckReader())
    assert stream.read(5) == b'hello'