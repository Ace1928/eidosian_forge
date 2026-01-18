import subprocess
import sys
import unittest
from datetime import datetime
from io import BytesIO
from testtools import TestCase
from testtools.compat import _b
from testtools.testresult.doubles import ExtendedTestResult, StreamResult
import iso8601
import subunit
from subunit.test_results import make_tag_filter, TestResultFilter
from subunit import ByteStreamToStreamResult, StreamResultToBytes
def test_exclude_skips(self):
    filtered_result = subunit.TestResultStats(None)
    result_filter = TestResultFilter(filtered_result, filter_skip=True)
    self.run_tests(result_filter)
    self.assertEqual(0, filtered_result.skipped_tests)
    self.assertEqual(2, filtered_result.failed_tests)
    self.assertEqual(3, filtered_result.testsRun)