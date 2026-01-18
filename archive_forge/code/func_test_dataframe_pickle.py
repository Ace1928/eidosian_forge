import pickle
import numpy as np
import pytest
import modin.pandas as pd
from modin.config import PersistentPickle
from modin.tests.pandas.utils import create_test_dfs, df_equals
@pytest.mark.parametrize('modin_df', [pytest.param(modin_df), pytest.param(pd.DataFrame(), id='empty_df')])
def test_dataframe_pickle(modin_df, persistent):
    other = pickle.loads(pickle.dumps(modin_df))
    df_equals(modin_df, other)