from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def assert_called_with(_mock_self, *args, **kwargs):
    """assert that the mock was called with the specified arguments.

        Raises an AssertionError if the args and keyword args passed in are
        different to the last call to the mock."""
    self = _mock_self
    if self.call_args is None:
        expected = self._format_mock_call_signature(args, kwargs)
        raise AssertionError('Expected call: %s\nNot called' % (expected,))

    def _error_message(cause):
        msg = self._format_mock_failure_message(args, kwargs)
        if six.PY2 and cause is not None:
            msg = '%s\n%s' % (msg, str(cause))
        return msg
    expected = self._call_matcher((args, kwargs))
    actual = self._call_matcher(self.call_args)
    if expected != actual:
        cause = expected if isinstance(expected, Exception) else None
        six.raise_from(AssertionError(_error_message(cause)), cause)