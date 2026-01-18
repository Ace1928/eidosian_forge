from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
def parentInstance(self, node, cls):
    for n in reversed(self._parents[node]):
        if isinstance(n, cls):
            return n
    raise ValueError('{} has no parent of type {}'.format(node, cls))