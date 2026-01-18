from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_string_and_tuples(self):
    c = Categorical(np.array(['c', ('a', 'b'), ('b', 'a'), 'c'], dtype=object))
    expected_index = Index([('a', 'b'), ('b', 'a'), 'c'])
    assert c.categories.equals(expected_index)