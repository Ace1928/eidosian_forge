import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def test_atleast_2d_column_default():
    import warnings
    assert np.all(atleast_2d_column_default([1, 2, 3]) == [[1], [2], [3]])
    assert atleast_2d_column_default(1).shape == (1, 1)
    assert atleast_2d_column_default([1]).shape == (1, 1)
    assert atleast_2d_column_default([[1]]).shape == (1, 1)
    assert atleast_2d_column_default([[[1]]]).shape == (1, 1, 1)
    assert atleast_2d_column_default([1, 2, 3]).shape == (3, 1)
    assert atleast_2d_column_default([[1], [2], [3]]).shape == (3, 1)
    with warnings.catch_warnings() as w:
        warnings.filterwarnings('ignore', 'the matrix subclass', PendingDeprecationWarning)
        assert type(atleast_2d_column_default(np.matrix(1))) == np.ndarray
        assert w is None
    global have_pandas
    if have_pandas:
        assert type(atleast_2d_column_default(pandas.Series([1, 2]))) == np.ndarray
        assert type(atleast_2d_column_default(pandas.DataFrame([[1], [2]]))) == np.ndarray
        assert type(atleast_2d_column_default(pandas.Series([1, 2]), preserve_pandas=True)) == pandas.DataFrame
        assert type(atleast_2d_column_default(pandas.DataFrame([[1], [2]]), preserve_pandas=True)) == pandas.DataFrame
        s = pandas.Series([10, 11, 12], name='hi', index=['a', 'b', 'c'])
        df = atleast_2d_column_default(s, preserve_pandas=True)
        assert isinstance(df, pandas.DataFrame)
        assert np.all(df.columns == ['hi'])
        assert np.all(df.index == ['a', 'b', 'c'])
    with warnings.catch_warnings() as w:
        warnings.filterwarnings('ignore', 'the matrix subclass', PendingDeprecationWarning)
        assert type(atleast_2d_column_default(np.matrix(1), preserve_pandas=True)) == np.ndarray
        assert w is None
    assert type(atleast_2d_column_default([1, 2, 3], preserve_pandas=True)) == np.ndarray
    if have_pandas:
        had_pandas = have_pandas
        try:
            have_pandas = False
            assert type(atleast_2d_column_default(pandas.Series([1, 2]), preserve_pandas=True)) == np.ndarray
            assert type(atleast_2d_column_default(pandas.DataFrame([[1], [2]]), preserve_pandas=True)) == np.ndarray
        finally:
            have_pandas = had_pandas