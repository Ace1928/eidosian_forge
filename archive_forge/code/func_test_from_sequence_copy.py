from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_sequence_copy(self):
    cat = Categorical(np.arange(5).repeat(2))
    result = Categorical._from_sequence(cat, dtype=cat.dtype, copy=False)
    assert result._codes is cat._codes
    result = Categorical._from_sequence(cat, dtype=cat.dtype, copy=True)
    assert not tm.shares_memory(result, cat)