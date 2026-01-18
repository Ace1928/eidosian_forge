from __future__ import unicode_literals
import collections
import contextlib
import inspect
import logging
import pprint
import sys
import textwrap
import six
def pop_section(self):
    self._dict_stack.pop(-1)