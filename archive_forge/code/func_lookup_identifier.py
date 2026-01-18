from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
def lookup_identifier(self, name):
    for d in reversed(self._definitions):
        if name in d:
            return d[name]
    return []