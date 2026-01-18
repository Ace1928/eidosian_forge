import numpy as np
import six
from patsy import PatsyError
from patsy.util import (SortAnythingKey,
def test_CategoricalSniffer():
    from patsy.missing import NAAction

    def t(NA_types, datas, exp_finish_fast, exp_levels, exp_contrast=None):
        sniffer = CategoricalSniffer(NAAction(NA_types=NA_types))
        for data in datas:
            done = sniffer.sniff(data)
            if done:
                assert exp_finish_fast
                break
            else:
                assert not exp_finish_fast
        assert sniffer.levels_contrast() == (exp_levels, exp_contrast)
    if have_pandas_categorical:
        preps = [lambda x: x, C]
        if have_pandas_categorical_dtype:
            preps += [pandas.Series, lambda x: C(pandas.Series(x))]
        for prep in preps:
            t([], [prep(pandas.Categorical([1, 2, None]))], True, (1, 2))
            t([], [prep(pandas_Categorical_from_codes([1, 0], ['a', 'b']))], True, ('a', 'b'))
            t([], [prep(pandas_Categorical_from_codes([1, 0], ['b', 'a']))], True, ('b', 'a'))
            obj = prep(pandas.Categorical(['a', 'b']))
            obj.contrast = 'CONTRAST'
            t([], [obj], True, ('a', 'b'), 'CONTRAST')
    t([], [C([1, 2]), C([3, 2])], False, (1, 2, 3))
    t([], [C([1, 2], levels=[1, 2, 3]), C([4, 2])], True, (1, 2, 3))
    t([], [C([1, 2], levels=[3, 2, 1]), C([4, 2])], True, (3, 2, 1))
    t(['None', 'NaN'], [C([1, np.nan]), C([10, None])], False, (1, 10))
    sniffer = CategoricalSniffer(NAAction(NA_types=['NaN']))
    sniffer.sniff(C([1, np.nan, None]))
    levels, _ = sniffer.levels_contrast()
    assert set(levels) == set([None, 1])
    t(['None', 'NaN'], [C([True, np.nan, None])], True, (False, True))
    t([], [C([10, 20]), C([False]), C([30, 40])], False, (False, True, 10, 20, 30, 40))
    t([], [np.asarray([True, False]), ['foo']], True, (False, True))
    t(['None', 'NaN'], [C([('b', 2), None, ('a', 1), np.nan, ('c', None)])], False, (('a', 1), ('b', 2), ('c', None)))
    t([], [C([10, 20], contrast='FOO')], False, (10, 20), 'FOO')
    t([], [[10, 30], [20]], False, (10, 20, 30))
    t([], [['b', 'a'], ['a']], False, ('a', 'b'))
    t([], ['b'], False, ('b',))
    import pytest
    sniffer = CategoricalSniffer(NAAction())
    pytest.raises(PatsyError, sniffer.sniff, [{}])
    pytest.raises(PatsyError, sniffer.sniff, np.asarray([['b']]))