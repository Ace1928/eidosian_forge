import operator
import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('method', [operator.methodcaller('to_period'), operator.methodcaller('tz_localize', 'CET'), operator.methodcaller('normalize'), operator.methodcaller('strftime', '%Y'), operator.methodcaller('round', 'h'), operator.methodcaller('floor', 'h'), operator.methodcaller('ceil', 'h'), operator.methodcaller('month_name'), operator.methodcaller('day_name')], ids=idfn)
def test_datetime_method(method):
    s = pd.Series(pd.date_range('2000', periods=4))
    s.attrs = {'a': 1}
    result = method(s.dt)
    assert result.attrs == {'a': 1}