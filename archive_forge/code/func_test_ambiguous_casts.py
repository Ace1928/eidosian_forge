from itertools import product, permutations
from collections import defaultdict
import unittest
from numba.core.base import OverloadSelector
from numba.core.registry import cpu_target
from numba.core.imputils import builtin_registry, RegistryLoader
from numba.core import types
from numba.core.errors import NumbaNotImplementedError, NumbaTypeError
def test_ambiguous_casts(self):
    os = self.create_overload_selector(kind='casts')
    all_types = set((t for sig, impl in os.versions for t in sig))
    for sig in permutations(all_types, r=2):
        try:
            os.find(sig)
        except NumbaNotImplementedError:
            pass