from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas.compat import (
from pandas.compat.numpy import np_version_lt1p23
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.from_dataframe import from_dataframe
from pandas.core.interchange.utils import ArrowCTypes
@pytest.mark.parametrize('tz', ['UTC', 'US/Pacific'])
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_datetimetzdtype(tz, unit):
    tz_data = pd.date_range('2018-01-01', periods=5, freq='D').tz_localize(tz).as_unit(unit)
    df = pd.DataFrame({'ts_tz': tz_data})
    tm.assert_frame_equal(df, from_dataframe(df.__dataframe__()))