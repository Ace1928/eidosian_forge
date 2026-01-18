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
def test_cotransform(self):
    with FugueWorkflow() as dag:
        a = dag.df([[1, 2], [1, 3], [2, 1]], 'a:int,b:int')
        b = dag.df([[1, 2], [3, 4]], 'a:int,c:int')
        c = dag.transform(a.zip(b), using=MockCoTransform1)
        e = dag.df([[1, 2, 1, 1]], 'a:int,ct1:int,ct2:int,p:int')
        e.assert_eq(c)
        c = dag.transform(a.zip(b), using=MockCoTransform1, params=dict(p=10))
        e = dag.df([[1, 2, 1, 10]], 'a:int,ct1:int,ct2:int,p:int')
        e.assert_eq(c)
        c = dag.transform(a.zip(b), using=mock_co_tf1, params=dict(p=10))
        e = dag.df([[1, 2, 1, 10]], 'a:int,ct1:int,ct2:int,p:int')
        e.assert_eq(c)
        a.zip(b).transform(mock_co_tf1, params=dict(p=10)).assert_eq(e)
        c = dag.transform(a.zip(b), using=mock_co_tf1, params=dict(p=10, col='x'))
        e = dag.df([[1, 2, 1, 10]], 'a:int,ct1:int,ct2:int,x:int')
        e.assert_eq(c)
        c = dag.transform(a.zip(b), using=mock_co_tf2, schema='a:int,ct1:int,ct2:int,p:int', params=dict(p=10))
        e = dag.df([[1, 2, 1, 10]], 'a:int,ct1:int,ct2:int,p:int')
        e.assert_eq(c)
        c = dag.transform(a.zip(), using=mock_co_tf3)
        e = dag.df([[1, 3, 1]], 'a:int,ct1:int,p:int')
        e.assert_eq(c)
        c = dag.transform(a.zip(partition=dict(by=['a'])), using=mock_co_tf3)
        e = dag.df([[1, 2, 1], [2, 1, 1]], 'a:int,ct1:int,p:int')
        e.assert_eq(c)
        c = a.partition_by('a').zip().transform(mock_co_tf3)
        e = dag.df([[1, 2, 1], [2, 1, 1]], 'a:int,ct1:int,p:int')
        e.assert_eq(c)
        c = a.partition(by=['a']).zip().transform(mock_co_tf4_ex, ignore_errors=[NotImplementedError])
        e = dag.df([[1, 2, 1]], 'a:int,ct1:int,p:int')
        e.assert_eq(c)
    dag.run(self.engine)