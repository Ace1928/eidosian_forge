from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def assert_called_once(_mock_self):
    """assert that the mock was called only once.
        """
    self = _mock_self
    if not self.call_count == 1:
        msg = "Expected '%s' to have been called once. Called %s times." % (self._mock_name or 'mock', self.call_count)
        raise AssertionError(msg)