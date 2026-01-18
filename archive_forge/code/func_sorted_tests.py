from collections import Counter
from pprint import pformat
from queue import Queue
import sys
import threading
import unittest
import testtools
def sorted_tests(suite_or_case, unpack_outer=False):
    """Sort suite_or_case while preserving non-vanilla TestSuites."""
    seen = Counter((case.id() for case in iterate_tests(suite_or_case)))
    duplicates = {test_id: count for test_id, count in seen.items() if count > 1}
    if duplicates:
        raise ValueError(f'Duplicate test ids detected: {pformat(duplicates)}')
    tests = _flatten_tests(suite_or_case, unpack_outer=unpack_outer)
    tests.sort()
    return unittest.TestSuite([test for sort_key, test in tests])