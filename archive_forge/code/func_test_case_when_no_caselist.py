import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_case_when_no_caselist(df):
    """
    Raise ValueError if no caselist is provided.
    """
    msg = 'provide at least one boolean condition, '
    msg += 'with a corresponding replacement.'
    with pytest.raises(ValueError, match=msg):
        df['a'].case_when([])