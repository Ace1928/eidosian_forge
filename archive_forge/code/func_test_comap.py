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
def test_comap(self):
    ps = PartitionSpec(presort='b,c')
    e = self.engine
    a = fa.as_fugue_engine_df(e, [[1, 2], [3, 4], [1, 5]], 'a:int,b:int')
    b = fa.as_fugue_engine_df(e, [[6, 1], [2, 7]], 'c:int,a:int')
    with raises(InvalidOperationError):
        e.zip(DataFrames([a, b]), partition_spec=PartitionSpec(by=['a']), how='cross')
    with raises(NotImplementedError):
        e.zip(DataFrames([a, b]), partition_spec=PartitionSpec(by=['a']), how='left_anti')
    z1 = fa.persist(e.zip(DataFrames([a, b])))
    z2 = fa.persist(e.zip(DataFrames([a, b]), partition_spec=ps, how='left_outer'))
    z3 = fa.persist(e.zip(DataFrames([b, a]), partition_spec=ps, how='right_outer'))
    z4 = fa.persist(e.zip(DataFrames([a, b]), partition_spec=ps, how='cross'))
    z5 = fa.persist(e.zip(DataFrames([a, b]), partition_spec=ps, how='full_outer'))

    def comap(cursor, dfs):
        assert not dfs.has_key
        v = ','.join([k + str(v.count()) for k, v in dfs.items()])
        keys = cursor.key_value_array if not dfs[0].empty else dfs[1][['a']].peek_array()
        if len(keys) == 0:
            return ArrayDataFrame([[v]], 'v:str')
        return ArrayDataFrame([keys + [v]], cursor.key_schema + 'v:str')

    def on_init(partition_no, dfs):
        assert not dfs.has_key
        assert partition_no >= 0
        assert len(dfs) > 0
    res = e.comap(z1, comap, 'a:int,v:str', PartitionSpec(), on_init=on_init)
    df_eq(res, [[1, '_02,_11']], 'a:int,v:str', throw=True)
    res = e.comap(z2, comap, 'a:int,v:str', PartitionSpec())
    df_eq(res, [[1, '_02,_11'], [3, '_01,_10']], 'a:int,v:str', throw=True)
    res = e.comap(z3, comap, 'a:int,v:str', PartitionSpec())
    df_eq(res, [[1, '_01,_12'], [3, '_00,_11']], 'a:int,v:str', throw=True)
    res = e.comap(z4, comap, 'v:str', PartitionSpec())
    df_eq(res, [['_03,_12']], 'v:str', throw=True)
    res = e.comap(z5, comap, 'a:int,v:str', PartitionSpec())
    df_eq(res, [[1, '_02,_11'], [3, '_01,_10'], [7, '_00,_11']], 'a:int,v:str', throw=True)