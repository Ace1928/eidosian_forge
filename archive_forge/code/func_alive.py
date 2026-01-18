from __future__ import absolute_import
import itertools
import sys
from weakref import ref
@property
def alive(self):
    """Whether finalizer is alive"""
    return self in self._registry