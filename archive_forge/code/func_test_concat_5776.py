import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.utils import from_pandas
from modin.utils import get_current_execution
from .utils import (
def test_concat_5776():
    modin_data = {key: pd.Series(index=range(3)) for key in ['a', 'b']}
    pandas_data = {key: pandas.Series(index=range(3)) for key in ['a', 'b']}
    df_equals(pd.concat(modin_data, axis='columns'), pandas.concat(pandas_data, axis='columns'))