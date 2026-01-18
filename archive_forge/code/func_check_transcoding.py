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
def check_transcoding(data, src_encoding, dest_encoding, chunk_sizes):
    chunk_sizes = iter(chunk_sizes)
    stream = pa.transcoding_input_stream(pa.BufferReader(data.encode(src_encoding)), src_encoding, dest_encoding)
    out = []
    while True:
        buf = stream.read(next(chunk_sizes))
        out.append(buf)
        if not buf:
            break
    out = b''.join(out)
    assert out.decode(dest_encoding) == data