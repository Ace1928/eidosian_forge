import datetime
import re
import sys
import threading
import time
import traceback
import unittest
from xml.sax import saxutils
from absl.testing import _pretty_print_reporter
def add_test_case_result(self, test_case_result):
    suite_name = type(test_case_result.test).__name__
    if suite_name == '_ErrorHolder':
        suite_name = test_case_result.full_class_name.rsplit('.')[-1]
    if isinstance(test_case_result.test, unittest.case._SubTest):
        suite_name = type(test_case_result.test.test_case).__name__
    self._setup_test_suite(suite_name)
    self.suites[suite_name].append(test_case_result)
    for error in test_case_result.errors:
        if error[0] == 'failure':
            self.failure_counts[suite_name] += 1
            break
        elif error[0] == 'error':
            self.error_counts[suite_name] += 1
            break