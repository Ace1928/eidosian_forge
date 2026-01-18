import datetime
import re
import sys
import threading
import time
import traceback
import unittest
from xml.sax import saxutils
from absl.testing import _pretty_print_reporter
def delete_pending_test_case_result(self, test):
    with self._pending_test_case_results_lock:
        test_id = id(test)
        del self.pending_test_case_results[test_id]