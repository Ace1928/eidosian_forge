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
def test_calling_time_None_enables_automatic_time(self):
    time = datetime.datetime(2009, 10, 11, 12, 13, 14, 15, iso8601.UTC)
    self.result.time(time)
    self.assertEqual(1, len(self.decorated._calls))
    self.assertEqual(time, self.decorated._calls[0])
    self.result.time(None)
    self.assertEqual(2, len(self.decorated._calls))
    self.assertEqual(None, self.decorated._calls[1])
    self.result.startTest(self)
    self.assertEqual(3, len(self.decorated._calls))
    self.assertNotEqual(None, self.decorated._calls[2])