import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def assertResultsMatch(self, test, result):
    events = list(result._events)
    self.assertEqual(('startTest', test), events.pop(0))
    for expected_result in test.expected_results:
        result = events.pop(0)
        if len(expected_result) == 1:
            self.assertEqual((expected_result[0], test), result)
        else:
            self.assertEqual((expected_result[0], test), result[:2])
            error_type = expected_result[1]
            self.assertIn(error_type.__name__, str(result[2]))
    self.assertEqual([('stopTest', test)], events)