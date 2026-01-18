import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.slow
@pytest.mark.parametrize('op, args, targop', [('cumprod', (), lambda x: x.cumprod()), ('cumsum', (), lambda x: x.cumsum()), ('shift', (-1,), lambda x: x.shift(-1)), ('shift', (1,), lambda x: x.shift())])
@pytest.mark.parametrize('df_fix', ['frame', 'frame_mi'])
@pytest.mark.parametrize('gb_target', [{'by': np.random.default_rng(2).integers(0, 50, size=10).astype(float)}, {'level': 0}, {'by': 'string'}, pytest.param({'by': 'string_missing'}, marks=pytest.mark.xfail), {'by': ['int', 'string']}])
def test_cython_transform_frame(request, op, args, targop, df_fix, gb_target):
    df = request.getfixturevalue(df_fix)
    gb = df.groupby(group_keys=False, **gb_target)
    if op != 'shift' and 'int' not in gb_target:
        i = gb[['int']].apply(targop)
        f = gb[['float', 'float_missing']].apply(targop)
        expected = concat([f, i], axis=1)
    else:
        if op != 'shift' or not isinstance(gb_target.get('by'), (str, list)):
            warn = None
        else:
            warn = DeprecationWarning
        msg = 'DataFrameGroupBy.apply operated on the grouping columns'
        with tm.assert_produces_warning(warn, match=msg):
            expected = gb.apply(targop)
    expected = expected.sort_index(axis=1)
    if op == 'shift':
        depr_msg = "The 'downcast' keyword in fillna is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            expected['string_missing'] = expected['string_missing'].fillna(np.nan, downcast=False)
            expected['string'] = expected['string'].fillna(np.nan, downcast=False)
    result = gb[expected.columns].transform(op, *args).sort_index(axis=1)
    tm.assert_frame_equal(result, expected)
    result = getattr(gb[expected.columns], op)(*args).sort_index(axis=1)
    tm.assert_frame_equal(result, expected)