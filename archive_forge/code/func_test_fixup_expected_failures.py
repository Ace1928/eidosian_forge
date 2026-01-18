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
def test_fixup_expected_failures(self):
    filtered_result = unittest.TestResult()
    result_filter = TestResultFilter(filtered_result, fixup_expected_failures={'failed'})
    self.run_tests(result_filter)
    self.assertEqual(['failed', 'todo'], [failure[0].id() for failure in filtered_result.expectedFailures])
    self.assertEqual([], filtered_result.failures)
    self.assertEqual(4, filtered_result.testsRun)