from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def decorate_class(self, klass):
    for attr in dir(klass):
        attr_value = getattr(klass, attr)
        if attr.startswith(patch.TEST_PREFIX) and hasattr(attr_value, '__call__'):
            decorator = _patch_dict(self.in_dict, self.values, self.clear)
            decorated = decorator(attr_value)
            setattr(klass, attr, decorated)
    return klass