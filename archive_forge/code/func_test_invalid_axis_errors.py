import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.utils import from_pandas
from modin.utils import get_current_execution
from .utils import (
def test_invalid_axis_errors():
    df, df2 = generate_dfs()
    modin_df, modin_df2 = (from_pandas(df), from_pandas(df2))
    with pytest.raises(ValueError):
        pd.concat([modin_df, modin_df2], axis=2)