from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_str_unknown(self):
    with pytest.raises(ValueError, match='Unknown dtype'):
        Categorical([1, 2], dtype='foo')