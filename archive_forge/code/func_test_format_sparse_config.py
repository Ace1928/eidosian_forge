import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_format_sparse_config(idx):
    msg = 'MultiIndex.format is deprecated'
    with pd.option_context('display.multi_sparse', False):
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = idx.format()
    assert result[1] == 'foo  two'