import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import ensure_clean_store
from pandas.io.pytables import read_hdf
def test_complex_across_dimensions_fixed(tmp_path, setup_path):
    complex128 = np.array([1.0 + 1j, 1.0 + 1j, 1.0 + 1j, 1.0 + 1j])
    s = Series(complex128, index=list('abcd'))
    df = DataFrame({'A': s, 'B': s})
    objs = [s, df]
    comps = [tm.assert_series_equal, tm.assert_frame_equal]
    for obj, comp in zip(objs, comps):
        path = tmp_path / setup_path
        obj.to_hdf(path, key='obj', format='fixed')
        reread = read_hdf(path, 'obj')
        comp(obj, reread)