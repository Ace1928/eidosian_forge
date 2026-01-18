from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def assert_not_called(_mock_self):
    """assert that the mock was never called.
        """
    self = _mock_self
    if self.call_count != 0:
        msg = "Expected '%s' to not have been called. Called %s times." % (self._mock_name or 'mock', self.call_count)
        raise AssertionError(msg)