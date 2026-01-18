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
def test_map_with_dict_col(self):
    e = self.engine
    dt = datetime.now()
    o = PandasDataFrame([[dt, dict(a=1)]], 'a:datetime,b:{a:long}')
    c = e.map_engine.map_dataframe(o, select_top, o.schema, PartitionSpec(by=['a']))
    df_eq(c, o, no_pandas=True, check_order=True, throw=True)

    def mp2(cursor, data):
        return data[['a']]
    c = e.map_engine.map_dataframe(o, mp2, 'a:datetime', PartitionSpec(by=['a']))
    df_eq(c, PandasDataFrame([[dt]], 'a:datetime'), no_pandas=True, check_order=True, throw=True)

    def mp3(cursor, data):
        return PandasDataFrame([[dt, dict(a=1)]], 'a:datetime,b:{a:long}')
    c = e.map_engine.map_dataframe(c, mp3, 'a:datetime,b:{a:long}', PartitionSpec(by=['a']))
    df_eq(c, o, no_pandas=True, check_order=True, throw=True)