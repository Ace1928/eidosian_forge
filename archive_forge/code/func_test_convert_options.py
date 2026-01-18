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
def test_convert_options(pickle_module):
    cls = ConvertOptions
    opts = cls()
    check_options_class(cls, check_utf8=[True, False], strings_can_be_null=[False, True], quoted_strings_can_be_null=[True, False], decimal_point=['.', ','], include_columns=[[], ['def', 'abc']], include_missing_columns=[False, True], auto_dict_encode=[False, True], timestamp_parsers=[[], [ISO8601, '%y-%m']])
    check_options_class_pickling(cls, pickler=pickle_module, check_utf8=False, strings_can_be_null=True, quoted_strings_can_be_null=False, decimal_point=',', include_columns=['def', 'abc'], include_missing_columns=False, auto_dict_encode=True, timestamp_parsers=[ISO8601, '%y-%m'])
    with pytest.raises(ValueError):
        opts.decimal_point = '..'
    assert opts.auto_dict_max_cardinality > 0
    opts.auto_dict_max_cardinality = 99999
    assert opts.auto_dict_max_cardinality == 99999
    assert opts.column_types == {}
    opts.column_types = {'b': pa.int16(), 'c': pa.float32()}
    assert opts.column_types == {'b': pa.int16(), 'c': pa.float32()}
    opts.column_types = {'v': 'int16', 'w': 'null'}
    assert opts.column_types == {'v': pa.int16(), 'w': pa.null()}
    schema = pa.schema([('a', pa.int32()), ('b', pa.string())])
    opts.column_types = schema
    assert opts.column_types == {'a': pa.int32(), 'b': pa.string()}
    opts.column_types = [('x', pa.binary())]
    assert opts.column_types == {'x': pa.binary()}
    with pytest.raises(TypeError, match='DataType expected'):
        opts.column_types = {'a': None}
    with pytest.raises(TypeError):
        opts.column_types = 0
    assert isinstance(opts.null_values, list)
    assert '' in opts.null_values
    assert 'N/A' in opts.null_values
    opts.null_values = ['xxx', 'yyy']
    assert opts.null_values == ['xxx', 'yyy']
    assert isinstance(opts.true_values, list)
    opts.true_values = ['xxx', 'yyy']
    assert opts.true_values == ['xxx', 'yyy']
    assert isinstance(opts.false_values, list)
    opts.false_values = ['xxx', 'yyy']
    assert opts.false_values == ['xxx', 'yyy']
    assert opts.timestamp_parsers == []
    opts.timestamp_parsers = [ISO8601]
    assert opts.timestamp_parsers == [ISO8601]
    opts = cls(column_types={'a': pa.null()}, null_values=['N', 'nn'], true_values=['T', 'tt'], false_values=['F', 'ff'], auto_dict_max_cardinality=999, timestamp_parsers=[ISO8601, '%Y-%m-%d'])
    assert opts.column_types == {'a': pa.null()}
    assert opts.null_values == ['N', 'nn']
    assert opts.false_values == ['F', 'ff']
    assert opts.true_values == ['T', 'tt']
    assert opts.auto_dict_max_cardinality == 999
    assert opts.timestamp_parsers == [ISO8601, '%Y-%m-%d']