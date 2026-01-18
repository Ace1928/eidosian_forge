from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_codes_with_dtype_raises(self):
    msg = 'Cannot specify'
    with pytest.raises(ValueError, match=msg):
        Categorical.from_codes([0, 1], categories=['a', 'b'], dtype=CategoricalDtype(['a', 'b']))
    with pytest.raises(ValueError, match=msg):
        Categorical.from_codes([0, 1], ordered=True, dtype=CategoricalDtype(['a', 'b']))