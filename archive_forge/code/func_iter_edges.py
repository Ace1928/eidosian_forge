import itertools
from .compat import collections_abc
def iter_edges(self):
    for f, children in self._forwards.items():
        for t in children:
            yield (f, t)