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
def testRetentionInSecondsMatch(self):
    secs = '30s'
    secs_match = RetentionInSecondsMatch(secs)
    self.assertEqual('30', secs_match.group('number'))
    secs = '1s'
    secs_match = RetentionInSecondsMatch(secs)
    self.assertEqual('1', secs_match.group('number'))
    secs = '1second'
    secs_match = RetentionInSecondsMatch(secs)
    self.assertEqual(None, secs_match)