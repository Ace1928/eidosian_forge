import copy
import os
import pickle
from datetime import datetime
from unittest import TestCase
import pandas as pd
import pytest
from pytest import raises
from triad.exceptions import InvalidOperationError
from triad.utils.io import isfile, makedirs, touch
import fugue.api as fa
import fugue.column.functions as ff
from fugue import (
from fugue.column import all_cols, col, lit
from fugue.dataframe.utils import _df_eq as df_eq
from fugue.execution.native_execution_engine import NativeExecutionEngine
def test_load_csv_folder(self):
    native = NativeExecutionEngine()
    a = ArrayDataFrame([[6.1, 1.1]], 'c:double,a:double')
    b = ArrayDataFrame([[2.1, 7.1], [4.1, 8.1]], 'c:double,a:double')
    path = os.path.join(self.tmpdir, 'a', 'b')
    fa.save(a, os.path.join(path, 'a.csv'), format_hint='csv', header=True, engine=native)
    fa.save(b, os.path.join(path, 'b.csv'), format_hint='csv', header=True, engine=native)
    touch(os.path.join(path, '_SUCCESS'))
    c = fa.load(path, format_hint='csv', header=True, infer_schema=True, columns=['a', 'c'], as_fugue=True)
    df_eq(c, [[1.1, 6.1], [7.1, 2.1], [8.1, 4.1]], 'a:double,c:double', throw=True)