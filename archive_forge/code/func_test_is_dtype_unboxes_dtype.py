import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import (
def test_is_dtype_unboxes_dtype(self, data, dtype):
    assert dtype.is_dtype(data) is True