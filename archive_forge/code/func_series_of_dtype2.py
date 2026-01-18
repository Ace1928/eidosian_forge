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
@pytest.fixture(params=get_series(), ids=lambda x: x.dtype.name)
def series_of_dtype2(request):
    """
    A duplicate of the series_of_dtype fixture, so that it can be used
    twice by a single function
    """
    return request.param