from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def attach_mock(self, mock, attribute):
    """
        Attach a mock as an attribute of this one, replacing its name and
        parent. Calls to the attached mock will be recorded in the
        `method_calls` and `mock_calls` attributes of this one."""
    mock._mock_parent = None
    mock._mock_new_parent = None
    mock._mock_name = ''
    mock._mock_new_name = None
    setattr(self, attribute, mock)