import numpy as np
import six
from patsy import PatsyError
from patsy.util import (SortAnythingKey,
def test_categorical_to_int():
    import pytest
    from patsy.missing import NAAction
    if have_pandas:
        s = pandas.Series(['a', 'b', 'c'], index=[10, 20, 30])
        c_pandas = categorical_to_int(s, ('a', 'b', 'c'), NAAction())
        assert np.all(c_pandas == [0, 1, 2])
        assert np.all(c_pandas.index == [10, 20, 30])
        pytest.raises(PatsyError, categorical_to_int, pandas.DataFrame({10: s}), ('a', 'b', 'c'), NAAction())
    if have_pandas_categorical:
        constructors = [pandas_Categorical_from_codes]
        if have_pandas_categorical_dtype:

            def Series_from_codes(codes, categories):
                c = pandas_Categorical_from_codes(codes, categories)
                return pandas.Series(c)
            constructors.append(Series_from_codes)
        for con in constructors:
            cat = con([1, 0, -1], ('a', 'b'))
            conv = categorical_to_int(cat, ('a', 'b'), NAAction())
            assert np.all(conv == [1, 0, -1])
            cat2 = con([1, 0, -1], ('a', 'None'))
            conv2 = categorical_to_int(cat, ('a', 'b'), NAAction(NA_types=['None']))
            assert np.all(conv2 == [1, 0, -1])
            pytest.raises(PatsyError, categorical_to_int, con([1, 0], ('a', 'b')), ('a', 'c'), NAAction())
            pytest.raises(PatsyError, categorical_to_int, con([1, 0], ('a', 'b')), ('b', 'a'), NAAction())

    def t(data, levels, expected, NA_action=NAAction()):
        got = categorical_to_int(data, levels, NA_action)
        assert np.array_equal(got, expected)
    t(['a', 'b', 'a'], ('a', 'b'), [0, 1, 0])
    t(np.asarray(['a', 'b', 'a']), ('a', 'b'), [0, 1, 0])
    t(np.asarray(['a', 'b', 'a'], dtype=object), ('a', 'b'), [0, 1, 0])
    t([0, 1, 2], (1, 2, 0), [2, 0, 1])
    t(np.asarray([0, 1, 2]), (1, 2, 0), [2, 0, 1])
    t(np.asarray([0, 1, 2], dtype=float), (1, 2, 0), [2, 0, 1])
    t(np.asarray([0, 1, 2], dtype=object), (1, 2, 0), [2, 0, 1])
    t(['a', 'b', 'a'], ('a', 'd', 'z', 'b'), [0, 3, 0])
    t([('a', 1), ('b', 0), ('a', 1)], (('a', 1), ('b', 0)), [0, 1, 0])
    pytest.raises(PatsyError, categorical_to_int, ['a', 'b', 'a'], ('a', 'c'), NAAction())
    t(C(['a', 'b', 'a']), ('a', 'b'), [0, 1, 0])
    t(C(['a', 'b', 'a']), ('b', 'a'), [1, 0, 1])
    t(C(['a', 'b', 'a'], levels=['b', 'a']), ('b', 'a'), [1, 0, 1])
    pytest.raises(PatsyError, categorical_to_int, C(['a', 'b', 'a'], levels=['a', 'b']), ('b', 'a'), NAAction())
    t('a', ('a', 'b'), [0])
    t('b', ('a', 'b'), [1])
    t(True, (False, True), [1])
    pytest.raises(PatsyError, categorical_to_int, np.asarray([['a', 'b'], ['b', 'a']]), ('a', 'b'), NAAction())
    pytest.raises(PatsyError, categorical_to_int, ['a', 'b'], ('a', 'b', {}), NAAction())
    pytest.raises(PatsyError, categorical_to_int, ['a', 'b', {}], ('a', 'b'), NAAction())
    t(['b', None, np.nan, 'a'], ('a', 'b'), [1, -1, -1, 0], NAAction(NA_types=['None', 'NaN']))
    t(['b', None, np.nan, 'a'], ('a', 'b', None), [1, -1, -1, 0], NAAction(NA_types=['None', 'NaN']))
    t(['b', None, np.nan, 'a'], ('a', 'b', None), [1, 2, -1, 0], NAAction(NA_types=['NaN']))
    pytest.raises(PatsyError, categorical_to_int, ['a', 'b', 'q'], ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'), NAAction())