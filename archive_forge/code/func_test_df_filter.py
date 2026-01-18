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
def test_df_filter(self):
    with FugueWorkflow() as dag:
        a = dag.df([[1, 10], [2, 20], [3, 30]], 'x:int,y:int')
        b = dag.df([[2, 20]], 'x:int,y:int')
        a.filter((col('y') > 15) & (col('y') < 25)).assert_eq(b)
    dag.run(self.engine)