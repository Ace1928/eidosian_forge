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
def test_merge_indicator_arg_validation(self, dfs_for_indicator):
    df1, df2 = dfs_for_indicator
    msg = 'indicator option can only accept boolean or string arguments'
    with pytest.raises(ValueError, match=msg):
        merge(df1, df2, on='col1', how='outer', indicator=5)
    with pytest.raises(ValueError, match=msg):
        df1.merge(df2, on='col1', how='outer', indicator=5)