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
def test_comap_with_key(self):
    e = self.engine
    a = fa.as_fugue_engine_df(e, [[1, 2], [3, 4], [1, 5]], 'a:int,b:int')
    b = fa.as_fugue_engine_df(e, [[6, 1], [2, 7]], 'c:int,a:int')
    c = fa.as_fugue_engine_df(e, [[6, 1]], 'c:int,a:int')
    z1 = fa.persist(e.zip(DataFrames(x=a, y=b)))
    z2 = fa.persist(e.zip(DataFrames(x=a, y=b, z=b)))
    z3 = fa.persist(e.zip(DataFrames(z=c), partition_spec=PartitionSpec(by=['a'])))

    def comap(cursor, dfs):
        assert dfs.has_key
        v = ','.join([k + str(v.count()) for k, v in dfs.items()])
        keys = cursor.key_value_array
        return ArrayDataFrame([keys + [v]], cursor.key_schema + 'v:str')

    def on_init(partition_no, dfs):
        assert dfs.has_key
        assert partition_no >= 0
        assert len(dfs) > 0
    res = e.comap(z1, comap, 'a:int,v:str', PartitionSpec(), on_init=on_init)
    df_eq(res, [[1, 'x2,y1']], 'a:int,v:str', throw=True)
    res = e.comap(z2, comap, 'a:int,v:str', PartitionSpec(), on_init=on_init)
    df_eq(res, [[1, 'x2,y1,z1']], 'a:int,v:str', throw=True)
    res = e.comap(z3, comap, 'a:int,v:str', PartitionSpec(), on_init=on_init)
    df_eq(res, [[1, 'z1']], 'a:int,v:str', throw=True)