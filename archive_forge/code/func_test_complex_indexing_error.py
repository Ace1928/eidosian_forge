import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import ensure_clean_store
from pandas.io.pytables import read_hdf
def test_complex_indexing_error(setup_path):
    complex128 = np.array([1.0 + 1j, 1.0 + 1j, 1.0 + 1j, 1.0 + 1j], dtype=np.complex128)
    df = DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'b', 'c', 'd'], 'C': complex128}, index=list('abcd'))
    msg = 'Columns containing complex values can be stored but cannot be indexed when using table format. Either use fixed format, set index=False, or do not include the columns containing complex values to data_columns when initializing the table.'
    with ensure_clean_store(setup_path) as store:
        with pytest.raises(TypeError, match=msg):
            store.append('df', df, data_columns=['C'])