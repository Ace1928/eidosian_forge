import datetime
import re
import sys
import threading
import time
import traceback
import unittest
from xml.sax import saxutils
from absl.testing import _pretty_print_reporter
def get_pending_test_case_result(self, test):
    test_id = id(test)
    return self.pending_test_case_results.get(test_id, None)