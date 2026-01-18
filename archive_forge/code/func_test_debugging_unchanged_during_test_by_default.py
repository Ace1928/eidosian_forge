import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_debugging_unchanged_during_test_by_default(self):
    debugging = [(defer.Deferred.debug, DelayedCall.debug)]

    class SomeCase(TestCase):

        def test_debugging_enabled(self):
            debugging.append((defer.Deferred.debug, DelayedCall.debug))
    test = SomeCase('test_debugging_enabled')
    runner = AsynchronousDeferredRunTest(test, handlers=test.exception_handlers, reactor=self.make_reactor(), timeout=self.make_timeout())
    runner.run(self.make_result())
    self.assertEqual(debugging[0], debugging[1])