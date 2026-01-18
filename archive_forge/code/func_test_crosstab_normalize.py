import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_crosstab_normalize(self):
    df = DataFrame({'a': [1, 2, 2, 2, 2], 'b': [3, 3, 4, 4, 4], 'c': [1, 1, np.nan, 1, 1]})
    rindex = Index([1, 2], name='a')
    cindex = Index([3, 4], name='b')
    full_normal = DataFrame([[0.2, 0], [0.2, 0.6]], index=rindex, columns=cindex)
    row_normal = DataFrame([[1.0, 0], [0.25, 0.75]], index=rindex, columns=cindex)
    col_normal = DataFrame([[0.5, 0], [0.5, 1.0]], index=rindex, columns=cindex)
    tm.assert_frame_equal(crosstab(df.a, df.b, normalize='all'), full_normal)
    tm.assert_frame_equal(crosstab(df.a, df.b, normalize=True), full_normal)
    tm.assert_frame_equal(crosstab(df.a, df.b, normalize='index'), row_normal)
    tm.assert_frame_equal(crosstab(df.a, df.b, normalize='columns'), col_normal)
    tm.assert_frame_equal(crosstab(df.a, df.b, normalize=1), crosstab(df.a, df.b, normalize='columns'))
    tm.assert_frame_equal(crosstab(df.a, df.b, normalize=0), crosstab(df.a, df.b, normalize='index'))
    row_normal_margins = DataFrame([[1.0, 0], [0.25, 0.75], [0.4, 0.6]], index=Index([1, 2, 'All'], name='a', dtype='object'), columns=Index([3, 4], name='b', dtype='object'))
    col_normal_margins = DataFrame([[0.5, 0, 0.2], [0.5, 1.0, 0.8]], index=Index([1, 2], name='a', dtype='object'), columns=Index([3, 4, 'All'], name='b', dtype='object'))
    all_normal_margins = DataFrame([[0.2, 0, 0.2], [0.2, 0.6, 0.8], [0.4, 0.6, 1]], index=Index([1, 2, 'All'], name='a', dtype='object'), columns=Index([3, 4, 'All'], name='b', dtype='object'))
    tm.assert_frame_equal(crosstab(df.a, df.b, normalize='index', margins=True), row_normal_margins)
    tm.assert_frame_equal(crosstab(df.a, df.b, normalize='columns', margins=True), col_normal_margins)
    tm.assert_frame_equal(crosstab(df.a, df.b, normalize=True, margins=True), all_normal_margins)