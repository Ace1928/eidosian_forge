import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_case_when_caselist_is_not_a_list(df):
    """
    Raise ValueError if caselist is not a list.
    """
    msg = 'The caselist argument should be a list; '
    msg += 'instead got.+'
    with pytest.raises(TypeError, match=msg):
        df['a'].case_when(caselist=())