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
def test_invalid_row_handler(self, pickle_module):
    rows = b'a,b\nc\nd,e\nf,g,h\ni,j\n'
    parse_opts = ParseOptions()
    with pytest.raises(ValueError, match='Expected 2 columns, got 1: c'):
        self.read_bytes(rows, parse_options=parse_opts)
    parse_opts.invalid_row_handler = InvalidRowHandler('skip')
    table = self.read_bytes(rows, parse_options=parse_opts)
    assert table.to_pydict() == {'a': ['d', 'i'], 'b': ['e', 'j']}

    def row_num(x):
        return None if self.use_threads else x
    expected_rows = [InvalidRow(2, 1, row_num(2), 'c'), InvalidRow(2, 3, row_num(4), 'f,g,h')]
    assert parse_opts.invalid_row_handler.rows == expected_rows
    parse_opts.invalid_row_handler = InvalidRowHandler('error')
    with pytest.raises(ValueError, match='Expected 2 columns, got 1: c'):
        self.read_bytes(rows, parse_options=parse_opts)
    expected_rows = [InvalidRow(2, 1, row_num(2), 'c')]
    assert parse_opts.invalid_row_handler.rows == expected_rows
    parse_opts.invalid_row_handler = InvalidRowHandler('skip')
    parse_opts = pickle_module.loads(pickle_module.dumps(parse_opts))
    table = self.read_bytes(rows, parse_options=parse_opts)
    assert table.to_pydict() == {'a': ['d', 'i'], 'b': ['e', 'j']}