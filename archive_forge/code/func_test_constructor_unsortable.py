from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_unsortable(self):
    arr = np.array([1, 2, 3, datetime.now()], dtype='O')
    factor = Categorical(arr, ordered=False)
    assert not factor.ordered
    msg = "'values' is not ordered, please explicitly specify the categories order by passing in a categories argument."
    with pytest.raises(TypeError, match=msg):
        Categorical(arr, ordered=True)