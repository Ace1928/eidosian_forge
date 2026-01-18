import abc
import bz2
from datetime import date, datetime
from decimal import Decimal
import gc
import gzip
import io
import itertools
import os
import select
import shutil
import signal
import string
import tempfile
import threading
import time
import unittest
import weakref
import pytest
import numpy as np
import pyarrow as pa
from pyarrow.csv import (
from pyarrow.tests import util
def test_inference_failure(self):
    rows = b'a,b\n123,456\nabc,de\xff\ngh,ij\n'
    read_options = ReadOptions()
    read_options.block_size = len(rows) - 7
    reader = self.open_bytes(rows, read_options=read_options)
    expected_schema = pa.schema([('a', pa.int64()), ('b', pa.int64())])
    assert reader.schema == expected_schema
    assert reader.read_next_batch().to_pydict() == {'a': [123], 'b': [456]}
    with pytest.raises(ValueError, match='CSV conversion error to int64'):
        reader.read_next_batch()
    with pytest.raises(StopIteration):
        reader.read_next_batch()