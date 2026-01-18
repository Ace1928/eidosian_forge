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
@pytest.mark.parametrize('how', ['inner', 'left', 'right', 'outer'])
def test_merge_empty_dataframe(self, index, how):
    left = DataFrame([], index=index[:0])
    right = left.copy()
    result = left.join(right, how=how)
    tm.assert_frame_equal(result, left)