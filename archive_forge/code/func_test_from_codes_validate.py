from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('validate', [True, False])
def test_from_codes_validate(self, validate):
    dtype = CategoricalDtype(['a', 'b'])
    if validate:
        with pytest.raises(ValueError, match='codes need to be between '):
            Categorical.from_codes([4, 5], dtype=dtype, validate=validate)
    else:
        Categorical.from_codes([4, 5], dtype=dtype, validate=validate)