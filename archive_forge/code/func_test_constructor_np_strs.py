from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason="Can't be NumPy strings")
def test_constructor_np_strs(self):
    cat = Categorical(['1', '0', '1'], [np.str_('0'), np.str_('1')])
    assert all((isinstance(x, np.str_) for x in cat.categories))