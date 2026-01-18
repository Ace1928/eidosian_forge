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
def test_column_types(self):
    opts = ConvertOptions(column_types={'b': 'float32', 'c': 'string', 'd': 'boolean', 'e': pa.decimal128(11, 2), 'zz': 'null'})
    rows = b'a,b,c,d,e\n1,2,3,true,1.0\n4,-5,6,false,0\n'
    table = self.read_bytes(rows, convert_options=opts)
    schema = pa.schema([('a', pa.int64()), ('b', pa.float32()), ('c', pa.string()), ('d', pa.bool_()), ('e', pa.decimal128(11, 2))])
    expected = {'a': [1, 4], 'b': [2.0, -5.0], 'c': ['3', '6'], 'd': [True, False], 'e': [Decimal('1.00'), Decimal('0.00')]}
    assert table.schema == schema
    assert table.to_pydict() == expected
    opts = ConvertOptions(column_types=pa.schema([('b', pa.float32()), ('c', pa.string()), ('d', pa.bool_()), ('e', pa.decimal128(11, 2)), ('zz', pa.bool_())]))
    table = self.read_bytes(rows, convert_options=opts)
    assert table.schema == schema
    assert table.to_pydict() == expected
    rows = b'a,b,c,d,e\n1,XXX,3,true,5\n4,-5,6,false,7\n'
    with pytest.raises(pa.ArrowInvalid) as exc:
        self.read_bytes(rows, convert_options=opts)
    err = str(exc.value)
    assert 'In CSV column #1: ' in err
    assert "CSV conversion error to float: invalid value 'XXX'" in err