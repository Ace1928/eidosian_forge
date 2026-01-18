import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_elements_not_in_by_but_in_df(self):
    left = DataFrame([['g', 'h', 1], ['g', 'h', 3]], columns=list('GHE'))
    right = DataFrame([[2, 1]], columns=list('ET'))
    msg = "\\{'h'\\} not found in left columns"
    with pytest.raises(KeyError, match=msg):
        merge_ordered(left, right, on='E', left_by=['G', 'h'])