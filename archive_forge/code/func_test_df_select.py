import datetime
import os
import pickle
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional
from unittest import TestCase
from uuid import uuid4
from triad.utils.io import write_text, join
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from fsspec.implementations.local import LocalFileSystem
from pytest import raises
from triad import SerializableRLock
import fugue.api as fa
from fugue import (
from fugue.column import col
from fugue.column import functions as ff
from fugue.column import lit
from fugue.dataframe.utils import _df_eq as df_eq
from fugue.exceptions import (
@pytest.mark.skipif(not HAS_QPD, reason='qpd not working')
def test_df_select(self):
    with FugueWorkflow() as dag:
        a = dag.df([[1, 10], [2, 20], [3, 30]], 'x:int,y:int')
        a.select('*').assert_eq(a)
        b = dag.df([[1, 10, 11, 'x'], [2, 20, 22, 'x'], [3, 30, 33, 'x']], 'x:int,y:int,c:int,d:str')
        a.select('*', (col('x') + col('y')).cast('int32').alias('c'), lit('x', 'd')).assert_eq(b)
        a = dag.df([[1, 10], [2, 20], [1, 10]], 'x:int,y:int')
        b = dag.df([[1, 10], [2, 20]], 'x:int,y:int')
        a.select('*', distinct=True).assert_eq(b)
        a = dag.df([[1, 10], [1, 20], [3, 30]], 'x:int,y:int')
        b = dag.df([[1, 30], [3, 30]], 'x:int,y:int')
        a.select('x', ff.sum(col('y')).cast('int32')).assert_eq(b)
        a = dag.df([[1, 10], [1, 20], [3, 35], [3, 40]], 'x:int,y:int')
        b = dag.df([[3, 35]], 'x:int,z:int')
        a.select('x', ff.sum(col('y')).alias('z').cast('int32'), where=col('y') < 40, having=ff.sum(col('y')) > 30).assert_eq(b)
        b = dag.df([[65]], 'z:long')
        a.select(ff.sum(col('y')).alias('z').cast(int), where=col('y') < 40).show()
        raises(ValueError, lambda: a.select('*', 'x'))
    dag.run(self.engine)
    with FugueWorkflow() as dag:
        a = dag.df([[0]], 'a:long')
        b = dag.df([[0]], 'a:long')
        dag.select('select * from', a).assert_eq(b)
    dag.run(self.engine, {'fugue.sql.compile.ignore_case': True})