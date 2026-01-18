from __future__ import unicode_literals
import collections
import contextlib
import inspect
import logging
import pprint
import sys
import textwrap
import six
def serialize_docstring(docstring):
    if '\n' in docstring:
        return docstring.split('\n')
    return docstring