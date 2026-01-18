from datetime import datetime
from dateutil.tz.tz import tzlocal
import pytest
from pandas._libs.tslibs import (
from pandas.compat import (
from pandas.tseries.offsets import (
def test_compare_str(_offset):
    off = _get_offset(_offset)
    assert not off == 'infer'
    assert off != 'foo'