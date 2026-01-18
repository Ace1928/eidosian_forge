from datetime import date, datetime
from typing import Any
from unittest import TestCase
import numpy as np
import pandas as pd
from pytest import raises
import fugue.api as fi
from fugue.dataframe import ArrowDataFrame, DataFrame
from fugue.dataframe.utils import _df_eq as df_eq
from fugue.exceptions import FugueDataFrameOperationError, FugueDatasetEmptyError
def test_alter_columns_invalid(self):
    with raises(Exception):
        df = self.df([['1', 'x'], ['2', 'y'], ['3', None]], 'a:str,b:str')
        ndf = fi.alter_columns(df, 'b:int')
        fi.show(ndf)