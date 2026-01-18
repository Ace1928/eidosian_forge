import csv
import datetime
import sys
import unittest
from io import StringIO
import testtools
from testtools import TestCase
from testtools.content import TracebackContent, text_content
from testtools.testresult.doubles import ExtendedTestResult
import subunit
import iso8601
import subunit.test_results
def test_add_failure(self):
    self.result.startTest(self)
    try:
        self.fail('intentional failure')
    except self.failureException:
        failure = sys.exc_info()
    self.result.addFailure(self, failure)
    self.result.stopTest(self)
    self.assertCalled(status='failure', details={'traceback': TracebackContent(failure, self)})