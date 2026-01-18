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
def testRetentionInYearsMatch(self):
    years = '30y'
    years_match = RetentionInYearsMatch(years)
    self.assertEqual('30', years_match.group('number'))
    years = '1y'
    years_match = RetentionInYearsMatch(years)
    self.assertEqual('1', years_match.group('number'))
    years = '1year'
    years_match = RetentionInYearsMatch(years)
    self.assertEqual(None, years_match)