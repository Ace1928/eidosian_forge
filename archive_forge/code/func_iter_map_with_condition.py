from __future__ import absolute_import, unicode_literals
import itertools
import warnings
from abc import ABCMeta, abstractmethod
import six
from pybtex import textutils
from pybtex.utils import collect_iterable, deprecated
from pybtex import py3compat
def iter_map_with_condition():
    length = len(self)
    for index, child in enumerate(self.parts):
        if hasattr(child, 'map'):
            yield (child.map(f, condition) if condition(index, length) else child)
        else:
            yield (f(child) if condition(index, length) else child)