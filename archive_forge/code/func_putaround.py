from __future__ import generators
from bisect import bisect_right
import sys
import inspect, tokenize
import py
from types import ModuleType
def putaround(self, before='', after='', indent=' ' * 4):
    """ return a copy of the source object with
            'before' and 'after' wrapped around it.
        """
    before = Source(before)
    after = Source(after)
    newsource = Source()
    lines = [indent + line for line in self.lines]
    newsource.lines = before.lines + lines + after.lines
    return newsource