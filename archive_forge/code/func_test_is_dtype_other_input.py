import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import (
def test_is_dtype_other_input(self, dtype):
    assert dtype.is_dtype([1, 2, 3]) is False