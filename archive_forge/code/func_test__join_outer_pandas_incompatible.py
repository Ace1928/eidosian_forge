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
def test__join_outer_pandas_incompatible(self):
    e = self.engine
    a = fa.as_fugue_engine_df(e, [[1, '2'], [3, '4']], 'a:int,b:str')
    b = fa.as_fugue_engine_df(e, [[6, 1], [2, 7]], 'c:int,a:int')
    c = fa.join(a, b, how='left_OUTER', on=['a'])
    df_eq(c, [[1, '2', 6], [3, '4', None]], 'a:int,b:str,c:int', throw=True)
    c = fa.join(b, a, how='left_outer', on=['a'])
    df_eq(c, [[6, 1, '2'], [2, 7, None]], 'c:int,a:int,b:str', throw=True)
    a = fa.as_fugue_engine_df(e, [[1, '2'], [3, '4']], 'a:int,b:str')
    b = fa.as_fugue_engine_df(e, [[True, 1], [False, 7]], 'c:bool,a:int')
    c = fa.join(a, b, how='left_OUTER', on=['a'])
    df_eq(c, [[1, '2', True], [3, '4', None]], 'a:int,b:str,c:bool', throw=True)
    c = fa.join(b, a, how='left_outer', on=['a'])
    df_eq(c, [[True, 1, '2'], [False, 7, None]], 'c:bool,a:int,b:str', throw=True)