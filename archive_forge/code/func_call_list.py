from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def call_list(self):
    """For a call object that represents multiple calls, `call_list`
        returns a list of all the intermediate calls as well as the
        final call."""
    vals = []
    thing = self
    while thing is not None:
        if thing.from_kall:
            vals.append(thing)
        thing = thing.parent
    return _CallList(reversed(vals))