import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_debugging_enabled_during_test_with_debug_flag(self):
    self.patch(defer.Deferred, 'debug', False)
    self.patch(DelayedCall, 'debug', False)
    debugging = []

    class SomeCase(TestCase):

        def test_debugging_enabled(self):
            debugging.append((defer.Deferred.debug, DelayedCall.debug))
    test = SomeCase('test_debugging_enabled')
    runner = AsynchronousDeferredRunTest(test, handlers=test.exception_handlers, reactor=self.make_reactor(), timeout=self.make_timeout(), debug=True)
    runner.run(self.make_result())
    self.assertEqual([(True, True)], debugging)
    self.assertEqual(False, defer.Deferred.debug)
    self.assertEqual(False, defer.Deferred.debug)