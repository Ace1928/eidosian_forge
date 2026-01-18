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
def test_column_types_dict(self):
    column_types = [('a', pa.dictionary(pa.int32(), pa.utf8())), ('b', pa.dictionary(pa.int32(), pa.int64())), ('c', pa.dictionary(pa.int32(), pa.decimal128(11, 2))), ('d', pa.dictionary(pa.int32(), pa.large_utf8()))]
    opts = ConvertOptions(column_types=dict(column_types))
    rows = b'a,b,c,d\nabc,123456,1.0,zz\ndefg,123456,0.5,xx\nabc,N/A,1.0,xx\n'
    table = self.read_bytes(rows, convert_options=opts)
    schema = pa.schema(column_types)
    expected = {'a': ['abc', 'defg', 'abc'], 'b': [123456, 123456, None], 'c': [Decimal('1.00'), Decimal('0.50'), Decimal('1.00')], 'd': ['zz', 'xx', 'xx']}
    assert table.schema == schema
    assert table.to_pydict() == expected
    column_types[0] = ('a', pa.dictionary(pa.int8(), pa.utf8()))
    opts = ConvertOptions(column_types=dict(column_types))
    with pytest.raises(NotImplementedError):
        table = self.read_bytes(rows, convert_options=opts)