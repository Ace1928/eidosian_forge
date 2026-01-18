import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@classmethod
def reduced_dim_assert(cls, result, expected):
    """
        Assertion about results with 1 fewer dimension that self.obj
        """
    tm.assert_series_equal(result, expected, check_names=False)
    assert result.name is None