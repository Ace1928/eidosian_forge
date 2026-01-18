import collections
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_na_flags_int_categories(self):
    categories = list(range(10))
    labels = np.random.default_rng(2).integers(0, 10, 20)
    labels[::5] = -1
    cat = Categorical(labels, categories)
    repr(cat)
    tm.assert_numpy_array_equal(isna(cat), labels == -1)