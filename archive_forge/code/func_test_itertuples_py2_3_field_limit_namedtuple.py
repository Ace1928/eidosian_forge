import datetime
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('limit', [254, 255, 1024])
@pytest.mark.parametrize('index', [True, False])
def test_itertuples_py2_3_field_limit_namedtuple(self, limit, index):
    df = DataFrame([{f'foo_{i}': f'bar_{i}' for i in range(limit)}])
    result = next(df.itertuples(index=index))
    assert isinstance(result, tuple)
    assert hasattr(result, '_fields')