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
def test__join_semi(self):
    e = self.engine
    a = fa.as_fugue_engine_df(e, [[1, 2], [3, 4]], 'a:int,b:int')
    b = fa.as_fugue_engine_df(e, [[6, 1], [2, 7]], 'c:int,a:int')
    c = fa.join(a, b, how='semi', on=['a'])
    df_eq(c, [[1, 2]], 'a:int,b:int', throw=True)
    c = fa.semi_join(b, a)
    df_eq(c, [[6, 1]], 'c:int,a:int', throw=True)
    b = fa.as_fugue_engine_df(e, [], 'c:int,a:int')
    c = fa.join(a, b, how='semi', on=['a'])
    df_eq(c, [], 'a:int,b:int', throw=True)
    a = fa.as_fugue_engine_df(e, [], 'a:int,b:int')
    b = fa.as_fugue_engine_df(e, [], 'c:int,a:int')
    c = fa.join(a, b, how='semi', on=['a'])
    df_eq(c, [], 'a:int,b:int', throw=True)