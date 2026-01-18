import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import NPartitions
from .utils import (
def test_api_indexer():
    modin_df, pandas_df = create_test_dfs(test_data_values[0])
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=3)
    pandas_rolled = pandas_df.rolling(window=indexer)
    modin_rolled = modin_df.rolling(window=indexer)
    df_equals(modin_rolled.sum(), pandas_rolled.sum())