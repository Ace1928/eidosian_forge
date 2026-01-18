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
@pytest.mark.parametrize('gb_target', [{'by': np.random.default_rng(2).integers(0, 50, size=10).astype(float)}, {'level': 0}, {'by': 'string'}, {'by': ['int', 'string']}])
@pytest.mark.parametrize('column', ['float', 'float_missing', 'int', 'datetime', 'timedelta', 'string', 'string_missing'])
def test_cython_transform_frame_column(request, op, args, targop, df_fix, gb_target, column):
    df = request.getfixturevalue(df_fix)
    gb = df.groupby(group_keys=False, **gb_target)
    c = column
    if c not in ['float', 'int', 'float_missing'] and op != 'shift' and (not (c == 'timedelta' and op == 'cumsum')):
        msg = '|'.join(['does not support .* operations', '.* is not supported for object dtype', 'is not implemented for this dtype'])
        with pytest.raises(TypeError, match=msg):
            gb[c].transform(op)
        with pytest.raises(TypeError, match=msg):
            getattr(gb[c], op)()
    else:
        expected = gb[c].apply(targop)
        expected.name = c
        if c in ['string_missing', 'string']:
            depr_msg = "The 'downcast' keyword in fillna is deprecated"
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                expected = expected.fillna(np.nan, downcast=False)
        res = gb[c].transform(op, *args)
        tm.assert_series_equal(expected, res)
        res2 = getattr(gb[c], op)(*args)
        tm.assert_series_equal(expected, res2)