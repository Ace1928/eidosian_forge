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
def test_load_parquet_files(self):
    native = NativeExecutionEngine()
    a = ArrayDataFrame([[6, 1]], 'c:int,a:long')
    b = ArrayDataFrame([[2, 7], [4, 8]], 'c:int,a:long')
    path = os.path.join(self.tmpdir, 'a', 'b')
    f1 = os.path.join(path, 'a.parquet')
    f2 = os.path.join(path, 'b.parquet')
    fa.save(a, f1, engine=native)
    fa.save(b, f2, engine=native)
    c = fa.load([f1, f2], format_hint='parquet', columns=['a', 'c'], as_fugue=True)
    df_eq(c, [[1, 6], [7, 2], [8, 4]], 'a:long,c:int', throw=True)