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
def test_merge_overlap(self, left):
    merged = merge(left, left, on='key')
    exp_len = (left['key'].value_counts() ** 2).sum()
    assert len(merged) == exp_len
    assert 'v1_x' in merged
    assert 'v1_y' in merged