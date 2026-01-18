import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_coerce(self, mixed_index, float_index):
    self.check_coerce(mixed_index, Index([1.5, 2, 3, 4, 5]))
    self.check_coerce(float_index, Index(np.arange(5) * 2.5))
    result = Index(np.array(np.arange(5) * 2.5, dtype=object))
    assert result.dtype == object
    self.check_coerce(float_index, result.astype('float64'))