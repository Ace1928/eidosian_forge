import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@td.skip_if_installed('matplotlib')
def test_import_error_message():
    df = DataFrame({'A': [1, 2]})
    with pytest.raises(ImportError, match='matplotlib is required for plotting'):
        df.plot()