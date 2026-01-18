import numpy as np
import six
from patsy import PatsyError
from patsy.util import (SortAnythingKey,
def test_guess_categorical():
    if have_pandas_categorical:
        c = pandas.Categorical([1, 2, 3])
        assert guess_categorical(c)
        if have_pandas_categorical_dtype:
            assert guess_categorical(pandas.Series(c))
    assert guess_categorical(C([1, 2, 3]))
    assert guess_categorical([True, False])
    assert guess_categorical(['a', 'b'])
    assert guess_categorical(['a', 'b', np.nan])
    assert guess_categorical(['a', 'b', None])
    assert not guess_categorical([1, 2, 3])
    assert not guess_categorical([1, 2, 3, np.nan])
    assert not guess_categorical([1.0, 2.0, 3.0])
    assert not guess_categorical([1.0, 2.0, 3.0, np.nan])