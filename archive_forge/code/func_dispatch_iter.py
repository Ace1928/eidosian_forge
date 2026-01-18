from __future__ import annotations
from warnings import warn
import inspect
from .conflict import ordering, ambiguities, super_signature, AmbiguityWarning
from .utils import expand_tuples
import itertools as itl
def dispatch_iter(self, *types):
    n = len(types)
    for signature in self.ordering:
        if len(signature) == n and all(map(issubclass, types, signature)):
            result = self.funcs[signature]
            yield result