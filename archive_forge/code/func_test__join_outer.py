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
def test__join_outer(self):
    e = self.engine
    a = fa.as_fugue_engine_df(e, [], 'a:int,b:int')
    b = fa.as_fugue_engine_df(e, [], 'c:str,a:int')
    c = fa.left_outer_join(a, b)
    df_eq(c, [], 'a:int,b:int,c:str', throw=True)
    a = fa.as_fugue_engine_df(e, [], 'a:int,b:str')
    b = fa.as_fugue_engine_df(e, [], 'c:int,a:int')
    c = fa.right_outer_join(a, b)
    df_eq(c, [], 'a:int,b:str,c:int', throw=True)
    a = fa.as_fugue_engine_df(e, [], 'a:int,b:str')
    b = fa.as_fugue_engine_df(e, [], 'c:str,a:int')
    c = fa.full_outer_join(a, b)
    df_eq(c, [], 'a:int,b:str,c:str', throw=True)
    a = fa.as_fugue_engine_df(e, [[1, '2'], [3, '4']], 'a:int,b:str')
    b = fa.as_fugue_engine_df(e, [['6', 1], ['2', 7]], 'c:str,a:int')
    c = fa.join(a, b, how='left_OUTER', on=['a'])
    df_eq(c, [[1, '2', '6'], [3, '4', None]], 'a:int,b:str,c:str', throw=True)
    c = fa.join(b, a, how='left_outer', on=['a'])
    df_eq(c, [['6', 1, '2'], ['2', 7, None]], 'c:str,a:int,b:str', throw=True)
    a = fa.as_fugue_engine_df(e, [[1, '2'], [3, '4']], 'a:int,b:str')
    b = fa.as_fugue_engine_df(e, [[6, 1], [2, 7]], 'c:double,a:int')
    c = fa.join(a, b, how='left_OUTER', on=['a'])
    df_eq(c, [[1, '2', 6.0], [3, '4', None]], 'a:int,b:str,c:double', throw=True)
    c = fa.join(b, a, how='left_outer', on=['a'])
    df_eq(c, [[6.0, 1, '2'], [2.0, 7, None]], 'c:double,a:int,b:str', throw=True)
    a = fa.as_fugue_engine_df(e, [[1, '2'], [3, '4']], 'a:int,b:str')
    b = fa.as_fugue_engine_df(e, [['6', 1], ['2', 7]], 'c:str,a:int')
    c = fa.join(a, b, how='right_outer', on=['a'])
    df_eq(c, [[1, '2', '6'], [7, None, '2']], 'a:int,b:str,c:str', throw=True)
    c = fa.join(a, b, how='full_outer', on=['a'])
    df_eq(c, [[1, '2', '6'], [3, '4', None], [7, None, '2']], 'a:int,b:str,c:str', throw=True)