import calendar
from datetime import datetime
import locale
import unicodedata
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.tseries.frequencies import to_offset
def test_datetimeindex_accessors5(self):
    freq_m = to_offset('M')
    bm = to_offset('BM')
    qfeb = to_offset('Q-FEB')
    qsfeb = to_offset('QS-FEB')
    bq = to_offset('BQ')
    bqs_apr = to_offset('BQS-APR')
    as_nov = to_offset('AS-NOV')
    tests = [(freq_m.is_month_start(Timestamp('2013-06-01')), 1), (bm.is_month_start(Timestamp('2013-06-01')), 0), (freq_m.is_month_start(Timestamp('2013-06-03')), 0), (bm.is_month_start(Timestamp('2013-06-03')), 1), (qfeb.is_month_end(Timestamp('2013-02-28')), 1), (qfeb.is_quarter_end(Timestamp('2013-02-28')), 1), (qfeb.is_year_end(Timestamp('2013-02-28')), 1), (qfeb.is_month_start(Timestamp('2013-03-01')), 1), (qfeb.is_quarter_start(Timestamp('2013-03-01')), 1), (qfeb.is_year_start(Timestamp('2013-03-01')), 1), (qsfeb.is_month_end(Timestamp('2013-03-31')), 1), (qsfeb.is_quarter_end(Timestamp('2013-03-31')), 0), (qsfeb.is_year_end(Timestamp('2013-03-31')), 0), (qsfeb.is_month_start(Timestamp('2013-02-01')), 1), (qsfeb.is_quarter_start(Timestamp('2013-02-01')), 1), (qsfeb.is_year_start(Timestamp('2013-02-01')), 1), (bq.is_month_end(Timestamp('2013-06-30')), 0), (bq.is_quarter_end(Timestamp('2013-06-30')), 0), (bq.is_year_end(Timestamp('2013-06-30')), 0), (bq.is_month_end(Timestamp('2013-06-28')), 1), (bq.is_quarter_end(Timestamp('2013-06-28')), 1), (bq.is_year_end(Timestamp('2013-06-28')), 0), (bqs_apr.is_month_end(Timestamp('2013-06-30')), 0), (bqs_apr.is_quarter_end(Timestamp('2013-06-30')), 0), (bqs_apr.is_year_end(Timestamp('2013-06-30')), 0), (bqs_apr.is_month_end(Timestamp('2013-06-28')), 1), (bqs_apr.is_quarter_end(Timestamp('2013-06-28')), 1), (bqs_apr.is_year_end(Timestamp('2013-03-29')), 1), (as_nov.is_year_start(Timestamp('2013-11-01')), 1), (as_nov.is_year_end(Timestamp('2013-10-31')), 1), (Timestamp('2012-02-01').days_in_month, 29), (Timestamp('2013-02-01').days_in_month, 28)]
    for ts, value in tests:
        assert ts == value