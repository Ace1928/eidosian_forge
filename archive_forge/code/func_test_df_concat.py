import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.utils import from_pandas
from modin.utils import get_current_execution
from .utils import (
def test_df_concat():
    df, df2 = generate_dfs()
    df_equals(pd.concat([df, df2]), pandas.concat([df, df2]))