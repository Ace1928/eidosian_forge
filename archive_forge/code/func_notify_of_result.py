import gi
import sys
from gi.repository import Notify  # noqa: E402
from testtools import StreamToExtendedDecorator  # noqa: E402
from subunit import TestResultStats  # noqa: E402
from subunit.filters import run_filter_script  # noqa: E402
def notify_of_result(result):
    result = result.decorated
    if result.failed_tests > 0:
        summary = 'Test run failed'
    else:
        summary = 'Test run successful'
    body = 'Total tests: %d; Passed: %d; Failed: %d' % (result.total_tests, result.passed_tests, result.failed_tests)
    nw = Notify.Notification(summary, body)
    nw.show()