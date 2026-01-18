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
def testRetentionPeriodToString(self):
    retention_str = _RetentionPeriodToString(SECONDS_IN_DAY)
    self.assertRegex(retention_str, 'Duration: 1 Day\\(s\\)')
    retention_str = _RetentionPeriodToString(SECONDS_IN_DAY - 1)
    self.assertRegex(retention_str, 'Duration: 86399 Second\\(s\\)')
    retention_str = _RetentionPeriodToString(SECONDS_IN_DAY + 1)
    self.assertRegex(retention_str, 'Duration: 86401 Seconds \\(~1 Day\\(s\\)\\)')
    retention_str = _RetentionPeriodToString(SECONDS_IN_MONTH)
    self.assertRegex(retention_str, 'Duration: 1 Month\\(s\\)')
    retention_str = _RetentionPeriodToString(SECONDS_IN_MONTH - 1)
    self.assertRegex(retention_str, 'Duration: 2678399 Seconds \\(~30 Day\\(s\\)\\)')
    retention_str = _RetentionPeriodToString(SECONDS_IN_MONTH + 1)
    self.assertRegex(retention_str, 'Duration: 2678401 Seconds \\(~31 Day\\(s\\)\\)')
    retention_str = _RetentionPeriodToString(SECONDS_IN_YEAR)
    self.assertRegex(retention_str, 'Duration: 1 Year\\(s\\)')
    retention_str = _RetentionPeriodToString(SECONDS_IN_YEAR - 1)
    self.assertRegex(retention_str, 'Duration: 31557599 Seconds \\(~365 Day\\(s\\)\\)')
    retention_str = _RetentionPeriodToString(SECONDS_IN_YEAR + 1)
    self.assertRegex(retention_str, 'Duration: 31557601 Seconds \\(~365 Day\\(s\\)\\)')