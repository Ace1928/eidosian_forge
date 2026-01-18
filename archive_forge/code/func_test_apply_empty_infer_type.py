from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('ax', ['index', 'columns'])
@pytest.mark.parametrize('func', [lambda x: x, lambda x: x.mean()], ids=['identity', 'mean'])
@pytest.mark.parametrize('raw', [True, False])
@pytest.mark.parametrize('axis', [0, 1])
def test_apply_empty_infer_type(ax, func, raw, axis, engine, request):
    df = DataFrame(**{ax: ['a', 'b', 'c']})
    with np.errstate(all='ignore'):
        test_res = func(np.array([], dtype='f8'))
        is_reduction = not isinstance(test_res, np.ndarray)
        result = df.apply(func, axis=axis, engine=engine, raw=raw)
        if is_reduction:
            agg_axis = df._get_agg_axis(axis)
            assert isinstance(result, Series)
            assert result.index is agg_axis
        else:
            assert isinstance(result, DataFrame)