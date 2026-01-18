import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test_fillna_invalid_value(data):
    modin_df = pd.DataFrame(data)
    pytest.raises(TypeError, modin_df.fillna, [1, 2])
    pytest.raises(TypeError, modin_df.fillna, (1, 2))
    pytest.raises(TypeError, modin_df.iloc[:, 0].fillna, modin_df)