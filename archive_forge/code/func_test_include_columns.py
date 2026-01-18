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
def test_include_columns(self):
    rows = b'ab,cd\nef,gh\nij,kl\nmn,op\n'
    convert_options = ConvertOptions()
    convert_options.include_columns = ['ab']
    table = self.read_bytes(rows, convert_options=convert_options)
    self.check_names(table, ['ab'])
    assert table.to_pydict() == {'ab': ['ef', 'ij', 'mn']}
    convert_options.include_columns = ['cd', 'ab']
    table = self.read_bytes(rows, convert_options=convert_options)
    schema = pa.schema([('cd', pa.string()), ('ab', pa.string())])
    assert table.schema == schema
    assert table.to_pydict() == {'cd': ['gh', 'kl', 'op'], 'ab': ['ef', 'ij', 'mn']}
    convert_options.include_columns = ['xx', 'ab', 'yy']
    with pytest.raises(KeyError, match="Column 'xx' in include_columns does not exist in CSV file"):
        self.read_bytes(rows, convert_options=convert_options)