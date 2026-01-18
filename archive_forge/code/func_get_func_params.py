from __future__ import annotations
from warnings import warn
import inspect
from .conflict import ordering, ambiguities, super_signature, AmbiguityWarning
from .utils import expand_tuples
import itertools as itl
@classmethod
def get_func_params(cls, func):
    if hasattr(inspect, 'signature'):
        sig = inspect.signature(func)
        return itl.islice(sig.parameters.values(), 1, None)