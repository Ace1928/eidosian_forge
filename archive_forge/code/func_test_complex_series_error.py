import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import ensure_clean_store
from pandas.io.pytables import read_hdf
def test_complex_series_error(tmp_path, setup_path):
    complex128 = np.array([1.0 + 1j, 1.0 + 1j, 1.0 + 1j, 1.0 + 1j])
    s = Series(complex128, index=list('abcd'))
    msg = 'Columns containing complex values can be stored but cannot be indexed when using table format. Either use fixed format, set index=False, or do not include the columns containing complex values to data_columns when initializing the table.'
    path = tmp_path / setup_path
    with pytest.raises(TypeError, match=msg):
        s.to_hdf(path, key='obj', format='t')
    path = tmp_path / setup_path
    s.to_hdf(path, key='obj', format='t', index=False)
    reread = read_hdf(path, 'obj')
    tm.assert_series_equal(s, reread)