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
def test_filter_predicate_with_tags(self):
    """You can filter by predicate callbacks that accept tags"""
    filtered_result = unittest.TestResult()

    def filter_cb(test, outcome, err, details, tags):
        return outcome == 'success'
    result_filter = TestResultFilter(filtered_result, filter_predicate=filter_cb, filter_success=False)
    self.run_tests(result_filter)
    self.assertEqual(1, filtered_result.testsRun)