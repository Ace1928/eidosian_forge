import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import ensure_clean_store
from pandas.io.pytables import read_hdf
def test_complex_fixed(tmp_path, setup_path):
    df = DataFrame(np.random.default_rng(2).random((4, 5)).astype(np.complex64), index=list('abcd'), columns=list('ABCDE'))
    path = tmp_path / setup_path
    df.to_hdf(path, key='df')
    reread = read_hdf(path, 'df')
    tm.assert_frame_equal(df, reread)
    df = DataFrame(np.random.default_rng(2).random((4, 5)).astype(np.complex128), index=list('abcd'), columns=list('ABCDE'))
    path = tmp_path / setup_path
    df.to_hdf(path, key='df')
    reread = read_hdf(path, 'df')
    tm.assert_frame_equal(df, reread)