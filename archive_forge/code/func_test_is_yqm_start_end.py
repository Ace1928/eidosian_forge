from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import PerformanceWarning
from pandas import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import WeekDay
from pandas.tseries import offsets
from pandas.tseries.offsets import (
def test_is_yqm_start_end():
    freq_m = to_offset('ME')
    bm = to_offset('BME')
    qfeb = to_offset('QE-FEB')
    qsfeb = to_offset('QS-FEB')
    bq = to_offset('BQE')
    bqs_apr = to_offset('BQS-APR')
    as_nov = to_offset('YS-NOV')
    tests = [(freq_m.is_month_start(Timestamp('2013-06-01')), 1), (bm.is_month_start(Timestamp('2013-06-01')), 0), (freq_m.is_month_start(Timestamp('2013-06-03')), 0), (bm.is_month_start(Timestamp('2013-06-03')), 1), (qfeb.is_month_end(Timestamp('2013-02-28')), 1), (qfeb.is_quarter_end(Timestamp('2013-02-28')), 1), (qfeb.is_year_end(Timestamp('2013-02-28')), 1), (qfeb.is_month_start(Timestamp('2013-03-01')), 1), (qfeb.is_quarter_start(Timestamp('2013-03-01')), 1), (qfeb.is_year_start(Timestamp('2013-03-01')), 1), (qsfeb.is_month_end(Timestamp('2013-03-31')), 1), (qsfeb.is_quarter_end(Timestamp('2013-03-31')), 0), (qsfeb.is_year_end(Timestamp('2013-03-31')), 0), (qsfeb.is_month_start(Timestamp('2013-02-01')), 1), (qsfeb.is_quarter_start(Timestamp('2013-02-01')), 1), (qsfeb.is_year_start(Timestamp('2013-02-01')), 1), (bq.is_month_end(Timestamp('2013-06-30')), 0), (bq.is_quarter_end(Timestamp('2013-06-30')), 0), (bq.is_year_end(Timestamp('2013-06-30')), 0), (bq.is_month_end(Timestamp('2013-06-28')), 1), (bq.is_quarter_end(Timestamp('2013-06-28')), 1), (bq.is_year_end(Timestamp('2013-06-28')), 0), (bqs_apr.is_month_end(Timestamp('2013-06-30')), 0), (bqs_apr.is_quarter_end(Timestamp('2013-06-30')), 0), (bqs_apr.is_year_end(Timestamp('2013-06-30')), 0), (bqs_apr.is_month_end(Timestamp('2013-06-28')), 1), (bqs_apr.is_quarter_end(Timestamp('2013-06-28')), 1), (bqs_apr.is_year_end(Timestamp('2013-03-29')), 1), (as_nov.is_year_start(Timestamp('2013-11-01')), 1), (as_nov.is_year_end(Timestamp('2013-10-31')), 1), (Timestamp('2012-02-01').days_in_month, 29), (Timestamp('2013-02-01').days_in_month, 28)]
    for ts, value in tests:
        assert ts == value