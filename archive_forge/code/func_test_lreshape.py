import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from .utils import df_equals, test_data_values
def test_lreshape():
    data = pd.DataFrame({'hr1': [514, 573], 'hr2': [545, 526], 'team': ['Red Sox', 'Yankees'], 'year1': [2007, 2008], 'year2': [2008, 2008]})
    with warns_that_defaulting_to_pandas():
        df = pd.lreshape(data, {'year': ['year1', 'year2'], 'hr': ['hr1', 'hr2']})
        assert isinstance(df, pd.DataFrame)
    with pytest.raises(ValueError):
        pd.lreshape(data.to_numpy(), {'year': ['year1', 'year2'], 'hr': ['hr1', 'hr2']})