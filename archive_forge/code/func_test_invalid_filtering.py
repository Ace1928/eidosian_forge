import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import Term
def test_invalid_filtering(setup_path):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
    with ensure_clean_store(setup_path) as store:
        store.put('df', df, format='table')
        msg = 'unable to collapse Joint Filters'
        with pytest.raises(NotImplementedError, match=msg):
            store.select('df', "columns=['A'] | columns=['B']")
        with pytest.raises(NotImplementedError, match=msg):
            store.select('df', "columns=['A','B'] & columns=['C']")