from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_codes_nan_cat_included(self):
    with pytest.raises(ValueError, match='Categorical categories cannot be null'):
        Categorical.from_codes([0, 1, 2], categories=['a', 'b', np.nan])