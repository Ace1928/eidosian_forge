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
def binary_map(cursor, df):
    arr = df.as_array(type_safe=True)
    for i in range(len(arr)):
        obj = pickle.loads(arr[i][0])
        obj.data += 'x'
        arr[i][0] = pickle.dumps(obj)
    return ArrayDataFrame(arr, df.schema)