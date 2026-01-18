import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def make_runner(self, test, timeout=None, suppress_twisted_logging=True, store_twisted_logs=True):
    if timeout is None:
        timeout = self.make_timeout()
    return AsynchronousDeferredRunTest(test, test.exception_handlers, timeout=timeout, suppress_twisted_logging=suppress_twisted_logging, store_twisted_logs=store_twisted_logs)