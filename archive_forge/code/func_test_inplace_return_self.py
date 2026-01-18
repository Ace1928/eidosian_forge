from copy import deepcopy
import inspect
import pydoc
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._config.config import option_context
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_inplace_return_self(self):
    data = DataFrame({'a': ['foo', 'bar', 'baz', 'qux'], 'b': [0, 0, 1, 1], 'c': [1, 2, 3, 4]})

    def _check_f(base, f):
        result = f(base)
        assert result is None
    f = lambda x: x.set_index('a', inplace=True)
    _check_f(data.copy(), f)
    f = lambda x: x.reset_index(inplace=True)
    _check_f(data.set_index('a'), f)
    f = lambda x: x.drop_duplicates(inplace=True)
    _check_f(data.copy(), f)
    f = lambda x: x.sort_values('b', inplace=True)
    _check_f(data.copy(), f)
    f = lambda x: x.sort_index(inplace=True)
    _check_f(data.copy(), f)
    f = lambda x: x.fillna(0, inplace=True)
    _check_f(data.copy(), f)
    f = lambda x: x.replace(1, 0, inplace=True)
    _check_f(data.copy(), f)
    f = lambda x: x.rename({1: 'foo'}, inplace=True)
    _check_f(data.copy(), f)
    d = data.copy()['c']
    f = lambda x: x.reset_index(inplace=True, drop=True)
    _check_f(data.set_index('a')['c'], f)
    f = lambda x: x.fillna(0, inplace=True)
    _check_f(d.copy(), f)
    f = lambda x: x.replace(1, 0, inplace=True)
    _check_f(d.copy(), f)
    f = lambda x: x.rename({1: 'foo'}, inplace=True)
    _check_f(d.copy(), f)