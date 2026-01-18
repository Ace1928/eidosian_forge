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
def test_datetime_in_workflow(self):

    def t1(df: pd.DataFrame) -> pd.DataFrame:
        df['b'] = '2020-01-02'
        df['b'] = pd.to_datetime(df['b'])
        return df

    class T2(Transformer):

        def get_output_schema(self, df):
            return df.schema

        def transform(self, df):
            return PandasDataFrame(df.as_pandas())
    with FugueWorkflow() as dag:
        a = dag.df([['2020-01-01']], 'a:date').transform(t1)
        b = dag.df([[datetime.date(2020, 1, 1), datetime.datetime(2020, 1, 2)]], 'a:date,b:datetime')
        b.assert_eq(a, no_pandas=True)
        c = dag.df([['2020-01-01', '2020-01-01 00:00:00']], 'a:date,b:datetime')
        c.transform(T2).assert_eq(c)
        c.partition(by=['a']).transform(T2).assert_eq(c)
    dag.run(self.engine)