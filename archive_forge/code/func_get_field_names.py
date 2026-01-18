from __future__ import unicode_literals
import collections
import contextlib
import inspect
import logging
import pprint
import sys
import textwrap
import six
@classmethod
def get_field_names(cls):
    return [descr.name for descr in cls._field_registry]