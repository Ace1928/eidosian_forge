from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
def process_rule(self, a, b):
    """process a -> b rule"""
    if not a or isinstance(b, bool):
        return
    if isinstance(a, bool):
        return
    if (a, b) in self._rules_seen:
        return
    else:
        self._rules_seen.add((a, b))
    try:
        self._process_rule(a, b)
    except TautologyDetected:
        pass