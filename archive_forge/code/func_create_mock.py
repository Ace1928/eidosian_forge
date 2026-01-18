from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def create_mock(self):
    entry = self.name
    parent = self.parent
    m = parent._get_child_mock(name=entry, _new_name=entry, _new_parent=parent)
    setattr(parent, entry, m)
    _set_return_value(parent, m, entry)
    return m