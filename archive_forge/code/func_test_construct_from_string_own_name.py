import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import (
def test_construct_from_string_own_name(self, dtype):
    result = dtype.construct_from_string(dtype.name)
    assert type(result) is type(dtype)
    result = type(dtype).construct_from_string(dtype.name)
    assert type(result) is type(dtype)