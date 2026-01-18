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
def test_include_missing_columns(self):
    rows = b'ab,cd\nef,gh\nij,kl\nmn,op\n'
    read_options = ReadOptions()
    convert_options = ConvertOptions()
    convert_options.include_columns = ['xx', 'ab', 'yy']
    convert_options.include_missing_columns = True
    table = self.read_bytes(rows, read_options=read_options, convert_options=convert_options)
    schema = pa.schema([('xx', pa.null()), ('ab', pa.string()), ('yy', pa.null())])
    assert table.schema == schema
    assert table.to_pydict() == {'xx': [None, None, None], 'ab': ['ef', 'ij', 'mn'], 'yy': [None, None, None]}
    read_options.column_names = ['xx', 'yy']
    convert_options.include_columns = ['yy', 'cd']
    table = self.read_bytes(rows, read_options=read_options, convert_options=convert_options)
    schema = pa.schema([('yy', pa.string()), ('cd', pa.null())])
    assert table.schema == schema
    assert table.to_pydict() == {'yy': ['cd', 'gh', 'kl', 'op'], 'cd': [None, None, None, None]}
    convert_options.column_types = {'yy': pa.binary(), 'cd': pa.int32()}
    table = self.read_bytes(rows, read_options=read_options, convert_options=convert_options)
    schema = pa.schema([('yy', pa.binary()), ('cd', pa.int32())])
    assert table.schema == schema
    assert table.to_pydict() == {'yy': [b'cd', b'gh', b'kl', b'op'], 'cd': [None, None, None, None]}