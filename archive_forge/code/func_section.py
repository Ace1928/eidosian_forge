from __future__ import unicode_literals
import collections
import contextlib
import inspect
import logging
import pprint
import sys
import textwrap
import six
@contextlib.contextmanager
def section(self, key):
    self.push_section(key)
    yield None
    self.pop_section()