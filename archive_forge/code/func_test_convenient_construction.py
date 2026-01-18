import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_convenient_construction(self):
    reactor = object()
    timeout = object()
    handler = object()
    factory = AsynchronousDeferredRunTest.make_factory(reactor, timeout)
    runner = factory(self, [handler])
    self.assertIs(reactor, runner._reactor)
    self.assertIs(timeout, runner._timeout)
    self.assertIs(self, runner.case)
    self.assertEqual([handler], runner.handlers)