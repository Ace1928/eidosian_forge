from __future__ import absolute_import, unicode_literals
import itertools
import warnings
from abc import ABCMeta, abstractmethod
import six
from pybtex import textutils
from pybtex.utils import collect_iterable, deprecated
from pybtex import py3compat
@deprecated('0.19', 'use slicing instead')
def apply_to_start(self, f):
    return self.map(f, lambda index, length: index == 0)