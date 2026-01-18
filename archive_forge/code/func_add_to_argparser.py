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
def add_to_argparser(cls, argparser):
    if '\n' in cls.__doc__:
        title, description = cls.__doc__.split('\n', 1)
    else:
        title = cls.__doc__
        description = None
    optgroup = argparser.add_argument_group(title=title, description=description)
    for descr in cls._field_registry:
        if isinstance(descr, SubtreeDescriptor):
            descr.add_to_argparser(argparser)
        elif isinstance(descr, FieldDescriptor):
            descr.add_to_argparse(optgroup)