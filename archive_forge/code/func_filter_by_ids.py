from collections import Counter
from pprint import pformat
from queue import Queue
import sys
import threading
import unittest
import testtools
def filter_by_ids(suite_or_case, test_ids):
    """Remove tests from suite_or_case where their id is not in test_ids.

    :param suite_or_case: A test suite or test case.
    :param test_ids: Something that supports the __contains__ protocol.
    :return: suite_or_case, unless suite_or_case was a case that itself
        fails the predicate when it will return a new unittest.TestSuite with
        no contents.

    This helper exists to provide backwards compatibility with older versions
    of Python (currently all versions :)) that don't have a native
    filter_by_ids() method on Test(Case|Suite).

    For subclasses of TestSuite, filtering is done by:
        - attempting to call suite.filter_by_ids(test_ids)
        - if there is no method, iterating the suite and identifying tests to
          remove, then removing them from _tests, manually recursing into
          each entry.

    For objects with an id() method - TestCases, filtering is done by:
        - attempting to return case.filter_by_ids(test_ids)
        - if there is no such method, checking for case.id() in test_ids
          and returning case if it is, or TestSuite() if it is not.

    For anything else, it is not filtered - it is returned as-is.

    To provide compatibility with this routine for a custom TestSuite, just
    define a filter_by_ids() method that will return a TestSuite equivalent to
    the original minus any tests not in test_ids.
    Similarly to provide compatibility for a custom TestCase that does
    something unusual define filter_by_ids to return a new TestCase object
    that will only run test_ids that are in the provided container. If none
    would run, return an empty TestSuite().

    The contract for this function does not require mutation - each filtered
    object can choose to return a new object with the filtered tests. However
    because existing custom TestSuite classes in the wild do not have this
    method, we need a way to copy their state correctly which is tricky:
    thus the backwards-compatible code paths attempt to mutate in place rather
    than guessing how to reconstruct a new suite.
    """
    if hasattr(suite_or_case, 'filter_by_ids'):
        return suite_or_case.filter_by_ids(test_ids)
    if hasattr(suite_or_case, 'id'):
        if suite_or_case.id() in test_ids:
            return suite_or_case
        else:
            return unittest.TestSuite()
    if isinstance(suite_or_case, unittest.TestSuite):
        filtered = []
        for item in suite_or_case:
            filtered.append(filter_by_ids(item, test_ids))
        suite_or_case._tests[:] = filtered
    return suite_or_case