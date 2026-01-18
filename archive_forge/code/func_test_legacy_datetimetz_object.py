from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
def test_legacy_datetimetz_object(datapath):
    expected = DataFrame({'A': Timestamp('20130102', tz='US/Eastern').as_unit('ns'), 'B': Timestamp('20130603', tz='CET').as_unit('ns')}, index=range(5))
    with ensure_clean_store(datapath('io', 'data', 'legacy_hdf', 'datetimetz_object.h5'), mode='r') as store:
        result = store['df']
        tm.assert_frame_equal(result, expected)