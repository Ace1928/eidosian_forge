from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
@pytest.fixture(params=get_series_na(), ids=lambda x: x.dtype.name)
def series_of_dtype_all_na(request):
    """
    A parametrized fixture returning a variety of Series with all NA
    values
    """
    return request.param