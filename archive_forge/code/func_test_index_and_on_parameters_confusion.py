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
def test_index_and_on_parameters_confusion(self, df, df2):
    msg = "right_index parameter must be of type bool, not <class 'list'>"
    with pytest.raises(ValueError, match=msg):
        merge(df, df2, how='left', left_index=False, right_index=['key1', 'key2'])
    msg = "left_index parameter must be of type bool, not <class 'list'>"
    with pytest.raises(ValueError, match=msg):
        merge(df, df2, how='left', left_index=['key1', 'key2'], right_index=False)
    with pytest.raises(ValueError, match=msg):
        merge(df, df2, how='left', left_index=['key1', 'key2'], right_index=['key1', 'key2'])