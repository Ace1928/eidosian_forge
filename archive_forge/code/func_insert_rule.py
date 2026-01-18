from collections import defaultdict
from functools import total_ordering
import enum
def insert_rule(self, a, b, rel):
    self._forwards[a].insert(b, rel)
    self._callback(a, b, rel)
    self._backwards[b].add(a)
    self.propagate(a, b, rel)