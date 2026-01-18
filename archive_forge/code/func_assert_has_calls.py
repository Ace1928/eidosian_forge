from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def assert_has_calls(self, calls, any_order=False):
    """assert the mock has been called with the specified calls.
        The `mock_calls` list is checked for the calls.

        If `any_order` is False (the default) then the calls must be
        sequential. There can be extra calls before or after the
        specified calls.

        If `any_order` is True then the calls can be in any order, but
        they must all appear in `mock_calls`."""
    expected = [self._call_matcher(c) for c in calls]
    cause = expected if isinstance(expected, Exception) else None
    all_calls = _CallList((self._call_matcher(c) for c in self.mock_calls))
    if not any_order:
        if expected not in all_calls:
            six.raise_from(AssertionError('Calls not found.\nExpected: %r\nActual: %r' % (_CallList(calls), self.mock_calls)), cause)
        return
    all_calls = list(all_calls)
    not_found = []
    for kall in expected:
        try:
            all_calls.remove(kall)
        except ValueError:
            not_found.append(kall)
    if not_found:
        six.raise_from(AssertionError('%r not all found in call list' % (tuple(not_found),)), cause)