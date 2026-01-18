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
def test_map_with_binary(self):
    e = self.engine
    o = ArrayDataFrame([[pickle.dumps(BinaryObject('a'))], [pickle.dumps(BinaryObject('b'))]], 'a:bytes')
    c = e.map_engine.map_dataframe(o, binary_map, o.schema, PartitionSpec())
    expected = ArrayDataFrame([[pickle.dumps(BinaryObject('ax'))], [pickle.dumps(BinaryObject('bx'))]], 'a:bytes')
    df_eq(expected, c, no_pandas=True, check_order=True, throw=True)