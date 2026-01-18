from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_join_object_index(self):
    rng = date_range('1/1/2000', periods=10)
    idx = Index(['a', 'b', 'c', 'd'])
    result = rng.join(idx, how='outer')
    assert isinstance(result[0], Timestamp)