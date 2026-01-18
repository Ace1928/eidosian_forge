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
def test_decimal_point(self):
    parse_options = ParseOptions(delimiter=';')
    rows = b'a;b\n1.25;2,5\nNA;-3\n-4;NA'
    table = self.read_bytes(rows, parse_options=parse_options)
    schema = pa.schema([('a', pa.float64()), ('b', pa.string())])
    assert table.schema == schema
    assert table.to_pydict() == {'a': [1.25, None, -4.0], 'b': ['2,5', '-3', 'NA']}
    convert_options = ConvertOptions(decimal_point=',')
    table = self.read_bytes(rows, parse_options=parse_options, convert_options=convert_options)
    schema = pa.schema([('a', pa.string()), ('b', pa.float64())])
    assert table.schema == schema
    assert table.to_pydict() == {'a': ['1.25', 'NA', '-4'], 'b': [2.5, -3.0, None]}