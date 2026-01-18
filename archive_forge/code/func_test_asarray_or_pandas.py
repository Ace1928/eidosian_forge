import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def test_asarray_or_pandas():
    import warnings
    assert type(asarray_or_pandas([1, 2, 3])) is np.ndarray
    with warnings.catch_warnings() as w:
        warnings.filterwarnings('ignore', 'the matrix subclass', PendingDeprecationWarning)
        assert type(asarray_or_pandas(np.matrix([[1, 2, 3]]))) is np.ndarray
        assert type(asarray_or_pandas(np.matrix([[1, 2, 3]]), subok=True)) is np.matrix
        assert w is None
    a = np.array([1, 2, 3])
    assert asarray_or_pandas(a) is a
    a_copy = asarray_or_pandas(a, copy=True)
    assert np.array_equal(a, a_copy)
    a_copy[0] = 100
    assert not np.array_equal(a, a_copy)
    assert np.allclose(asarray_or_pandas([1, 2, 3], dtype=float), [1.0, 2.0, 3.0])
    assert asarray_or_pandas([1, 2, 3], dtype=float).dtype == np.dtype(float)
    a_view = asarray_or_pandas(a, dtype=a.dtype)
    a_view[0] = 99
    assert a[0] == 99
    global have_pandas
    if have_pandas:
        s = pandas.Series([1, 2, 3], name='A', index=[10, 20, 30])
        s_view1 = asarray_or_pandas(s)
        assert s_view1.name == 'A'
        assert np.array_equal(s_view1.index, [10, 20, 30])
        s_view1[10] = 101
        assert s[10] == 101
        s_copy = asarray_or_pandas(s, copy=True)
        assert s_copy.name == 'A'
        assert np.array_equal(s_copy.index, [10, 20, 30])
        assert np.array_equal(s_copy, s)
        s_copy[10] = 100
        assert not np.array_equal(s_copy, s)
        assert asarray_or_pandas(s, dtype=float).dtype == np.dtype(float)
        s_view2 = asarray_or_pandas(s, dtype=s.dtype)
        assert s_view2.name == 'A'
        assert np.array_equal(s_view2.index, [10, 20, 30])
        s_view2[10] = 99
        assert s[10] == 99
        df = pandas.DataFrame([[1, 2, 3]], columns=['A', 'B', 'C'], index=[10])
        df_view1 = asarray_or_pandas(df)
        df_view1.loc[10, 'A'] = 101
        assert np.array_equal(df_view1.columns, ['A', 'B', 'C'])
        assert np.array_equal(df_view1.index, [10])
        assert df.loc[10, 'A'] == 101
        df_copy = asarray_or_pandas(df, copy=True)
        assert np.array_equal(df_copy, df)
        assert np.array_equal(df_copy.columns, ['A', 'B', 'C'])
        assert np.array_equal(df_copy.index, [10])
        df_copy.loc[10, 'A'] = 100
        assert not np.array_equal(df_copy, df)
        df_converted = asarray_or_pandas(df, dtype=float)
        assert df_converted['A'].dtype == np.dtype(float)
        assert np.allclose(df_converted, df)
        assert np.array_equal(df_converted.columns, ['A', 'B', 'C'])
        assert np.array_equal(df_converted.index, [10])
        df_view2 = asarray_or_pandas(df, dtype=df['A'].dtype)
        assert np.array_equal(df_view2.columns, ['A', 'B', 'C'])
        assert np.array_equal(df_view2.index, [10])
        assert np.array_equal(df, df_view2)
        had_pandas = have_pandas
        try:
            have_pandas = False
            assert type(asarray_or_pandas(pandas.Series([1, 2, 3]))) is np.ndarray
            assert type(asarray_or_pandas(pandas.DataFrame([[1, 2, 3]]))) is np.ndarray
        finally:
            have_pandas = had_pandas