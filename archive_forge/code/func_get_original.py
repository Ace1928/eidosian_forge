from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def get_original(self):
    target = self.getter()
    name = self.attribute
    original = DEFAULT
    local = False
    try:
        original = target.__dict__[name]
    except (AttributeError, KeyError):
        original = getattr(target, name, DEFAULT)
    else:
        local = True
    if name in _builtins and isinstance(target, ModuleType):
        self.create = True
    if not self.create and original is DEFAULT:
        raise AttributeError('%s does not have the attribute %r' % (target, name))
    return (original, local)