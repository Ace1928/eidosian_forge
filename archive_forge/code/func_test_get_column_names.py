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
def test_get_column_names(self):
    df = self.to_native_df(pd.DataFrame([[0, 1, 2]], columns=['0', '1', '2']))
    assert fi.get_column_names(df) == ['0', '1', '2']