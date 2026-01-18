import pickle
import numpy as np
import pytest
import modin.pandas as pd
from modin.config import PersistentPickle
from modin.tests.pandas.utils import create_test_dfs, df_equals
def test_column_pickle(modin_column, modin_df, persistent):
    dmp = pickle.dumps(modin_column)
    other = pickle.loads(dmp)
    df_equals(modin_column.to_frame(), other.to_frame())
    if persistent:
        assert len(dmp) < len(pickle.dumps(modin_df))