import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_astype_float64_to_object(self):
    float_index = Index([0.0, 2.5, 5.0, 7.5, 10.0], dtype=np.float64)
    result = float_index.astype(object)
    assert result.equals(float_index)
    assert float_index.equals(result)
    assert isinstance(result, Index) and result.dtype == object