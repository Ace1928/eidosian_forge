from __future__ import absolute_import
import gslib.tests.testcase as testcase
from gslib.utils.retention_util import _RetentionPeriodToString
from gslib.utils.retention_util import DaysToSeconds
from gslib.utils.retention_util import MonthsToSeconds
from gslib.utils.retention_util import RetentionInDaysMatch
from gslib.utils.retention_util import RetentionInMonthsMatch
from gslib.utils.retention_util import RetentionInSeconds
from gslib.utils.retention_util import RetentionInSecondsMatch
from gslib.utils.retention_util import RetentionInYearsMatch
from gslib.utils.retention_util import SECONDS_IN_DAY
from gslib.utils.retention_util import SECONDS_IN_MONTH
from gslib.utils.retention_util import SECONDS_IN_YEAR
from gslib.utils.retention_util import YearsToSeconds
def testRetentionInSeconds(self):
    one_year = '1y'
    one_year_in_seconds = RetentionInSeconds(one_year)
    self.assertEqual(SECONDS_IN_YEAR, one_year_in_seconds)
    one_month = '1m'
    one_month_in_seconds = RetentionInSeconds(one_month)
    self.assertEqual(SECONDS_IN_MONTH, one_month_in_seconds)
    one_day = '1d'
    one_day_in_seconds = RetentionInSeconds(one_day)
    self.assertEqual(SECONDS_IN_DAY, one_day_in_seconds)
    one_second = '1s'
    one_second_in_seconds = RetentionInSeconds(one_second)
    self.assertEqual(1, one_second_in_seconds)