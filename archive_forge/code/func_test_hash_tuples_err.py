import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
@pytest.mark.parametrize('val', [5, 'foo', pd.Timestamp('20130101')])
def test_hash_tuples_err(val):
    msg = 'must be convertible to a list-of-tuples'
    with pytest.raises(TypeError, match=msg):
        hash_tuples(val)