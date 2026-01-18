import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import (
def test_array_type(self, data, dtype):
    assert dtype.construct_array_type() is type(data)