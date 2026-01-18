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
def test_no_time_from_progress(self):
    self.result.progress(1, subunit.PROGRESS_CUR)
    self.assertEqual(0, len(self.decorated._calls))