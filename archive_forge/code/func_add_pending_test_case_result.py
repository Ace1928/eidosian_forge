import datetime
import re
import sys
import threading
import time
import traceback
import unittest
from xml.sax import saxutils
from absl.testing import _pretty_print_reporter
def add_pending_test_case_result(self, test, error_summary=None, skip_reason=None):
    """Adds result information to a test case result which may still be running.

    If a result entry for the test already exists, add_pending_test_case_result
    will add error summary tuples and/or overwrite skip_reason for the result.
    If it does not yet exist, a result entry will be created.
    Note that a test result is considered to have been run and passed
    only if there are no errors or skip_reason.

    Args:
      test: A test method as defined by unittest
      error_summary: A 4-tuple with the following entries:
          1) a string identifier of either "failure" or "error"
          2) an exception_type
          3) an exception_message
          4) a string version of a sys.exc_info()-style tuple of values
             ('error', err[0], err[1], self._exc_info_to_string(err))
             If the length of errors is 0, then the test is either passed or
             skipped.
      skip_reason: a string explaining why the test was skipped
    """
    with self._pending_test_case_results_lock:
        test_id = id(test)
        if test_id not in self.pending_test_case_results:
            self.pending_test_case_results[test_id] = self._TEST_CASE_RESULT_CLASS(test)
        if error_summary:
            self.pending_test_case_results[test_id].errors.append(error_summary)
        if skip_reason:
            self.pending_test_case_results[test_id].skip_reason = skip_reason