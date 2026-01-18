import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test_fillna_invalid_method(data):
    modin_df = pd.DataFrame(data)
    with pytest.raises(ValueError):
        modin_df.fillna(method='ffil')