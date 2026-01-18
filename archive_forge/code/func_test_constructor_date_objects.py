from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_date_objects(self):
    v = date.today()
    cat = Categorical([v, v])
    assert cat.categories.dtype == object
    assert type(cat.categories[0]) is date