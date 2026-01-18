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
def test_header_column_names(self):
    rows = b'ab,cd\nef,gh\nij,kl\nmn,op\n'
    opts = ReadOptions()
    opts.column_names = ['x', 'y']
    table = self.read_bytes(rows, read_options=opts)
    self.check_names(table, ['x', 'y'])
    assert table.to_pydict() == {'x': ['ab', 'ef', 'ij', 'mn'], 'y': ['cd', 'gh', 'kl', 'op']}
    opts.skip_rows = 3
    table = self.read_bytes(rows, read_options=opts)
    self.check_names(table, ['x', 'y'])
    assert table.to_pydict() == {'x': ['mn'], 'y': ['op']}
    opts.skip_rows = 4
    table = self.read_bytes(rows, read_options=opts)
    self.check_names(table, ['x', 'y'])
    assert table.to_pydict() == {'x': [], 'y': []}
    opts.skip_rows = 5
    with pytest.raises(pa.ArrowInvalid):
        table = self.read_bytes(rows, read_options=opts)
    opts.skip_rows = 0
    opts.column_names = ['x', 'y', 'z']
    with pytest.raises(pa.ArrowInvalid, match='Expected 3 columns, got 2'):
        table = self.read_bytes(rows, read_options=opts)
    rows = b'abcd\n,,,,,\nij,kl\nmn,op\n'
    opts.skip_rows = 2
    opts.column_names = ['x', 'y']
    table = self.read_bytes(rows, read_options=opts)
    self.check_names(table, ['x', 'y'])
    assert table.to_pydict() == {'x': ['ij', 'mn'], 'y': ['kl', 'op']}