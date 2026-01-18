from __future__ import unicode_literals
import collections
import contextlib
import inspect
import logging
import pprint
import sys
import textwrap
import six
def push_section(self, key):
    key = key.replace('-', '_')
    if self._dict_stack:
        stacktop = self._dict_stack[-1]
    else:
        stacktop = self
    if key not in stacktop:
        stacktop[key] = {}
    newtop = stacktop[key]
    self._dict_stack.append(newtop)