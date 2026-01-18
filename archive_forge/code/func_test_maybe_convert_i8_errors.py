from itertools import permutations
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('breaks1, breaks2', permutations([date_range('20180101', periods=4), date_range('20180101', periods=4, tz='US/Eastern'), timedelta_range('0 days', periods=4)], 2), ids=lambda x: str(x.dtype))
@pytest.mark.parametrize('make_key', [IntervalIndex.from_breaks, lambda breaks: Interval(breaks[0], breaks[1]), lambda breaks: breaks, lambda breaks: breaks[0], list], ids=['IntervalIndex', 'Interval', 'Index', 'scalar', 'list'])
def test_maybe_convert_i8_errors(self, breaks1, breaks2, make_key):
    index = IntervalIndex.from_breaks(breaks1)
    key = make_key(breaks2)
    msg = f'Cannot index an IntervalIndex of subtype {breaks1.dtype} with values of dtype {breaks2.dtype}'
    msg = re.escape(msg)
    with pytest.raises(ValueError, match=msg):
        index._maybe_convert_i8(key)