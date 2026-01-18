import contextlib
import datetime as dt
import hashlib
import tempfile
import time
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import (
def test_create_table_index_data_columns_argument(setup_path):
    with ensure_clean_store(setup_path) as store:

        def col(t, column):
            return getattr(store.get_storer(t).table.cols, column)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        df['string'] = 'foo'
        df['string2'] = 'bar'
        store.append('f', df, data_columns=['string'])
        assert col('f', 'index').is_indexed is True
        assert col('f', 'string').is_indexed is True
        msg = "'Cols' object has no attribute 'string2'"
        with pytest.raises(AttributeError, match=msg):
            col('f', 'string2').is_indexed
        msg = 'column string2 is not a data_column.\nIn order to read column string2 you must reload the dataframe \ninto HDFStore and include string2 with the data_columns argument.'
        with pytest.raises(AttributeError, match=msg):
            store.create_table_index('f', columns=['string2'])