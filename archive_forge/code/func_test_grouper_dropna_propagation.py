import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('dropna', (False, True))
def test_grouper_dropna_propagation(dropna):
    df = pd.DataFrame({'A': [0, 0, 1, None], 'B': [1, 2, 3, None]})
    gb = df.groupby('A', dropna=dropna)
    assert gb._grouper.dropna == dropna