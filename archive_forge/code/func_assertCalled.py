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
def assertCalled(self, **kwargs):
    defaults = {'test': self, 'tags': set(), 'details': None, 'start_time': 0, 'stop_time': 1}
    defaults.update(kwargs)
    self.assertEqual([defaults], self.log)