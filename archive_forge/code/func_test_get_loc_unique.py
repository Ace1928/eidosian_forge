import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_loc_unique(self):
    cidx = CategoricalIndex(list('abc'))
    result = cidx.get_loc('b')
    assert result == 1