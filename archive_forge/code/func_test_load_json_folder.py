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
def test_load_json_folder(self):
    native = NativeExecutionEngine()
    a = ArrayDataFrame([[6, 1], [3, 4]], 'c:int,a:long')
    b = ArrayDataFrame([[2, 7], [4, 8]], 'c:int,a:long')
    path = os.path.join(self.tmpdir, 'a', 'b')
    fa.save(a, os.path.join(path, 'a.json'), format_hint='json', engine=native)
    fa.save(b, os.path.join(path, 'b.json'), format_hint='json', engine=native)
    touch(os.path.join(path, '_SUCCESS'))
    c = fa.load(path, format_hint='json', columns=['a', 'c'], as_fugue=True)
    df_eq(c, [[1, 6], [7, 2], [8, 4], [4, 3]], 'a:long,c:long', throw=True)