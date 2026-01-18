import doctest
from pprint import pformat
import unittest
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import DocTestMatches, Equals
from testtools.testresult.doubles import StreamResult as LoggingStream
from testtools.testsuite import FixtureSuite, sorted_tests
from testtools.tests.helpers import LoggingResult
def test_broken_test(self):
    log = []

    def on_test(test, status, start_time, stop_time, tags, details):
        log.append((test.id(), status, set(details.keys())))

    class BrokenTest:

        def __call__(self):
            pass
        run = __call__
    original_suite = unittest.TestSuite([BrokenTest()])
    suite = ConcurrentTestSuite(original_suite, self.split_suite)
    suite.run(TestByTestResult(on_test))
    self.assertEqual([('broken-runner', 'error', {'traceback'})], log)