from itertools import product, permutations
from collections import defaultdict
import unittest
from numba.core.base import OverloadSelector
from numba.core.registry import cpu_target
from numba.core.imputils import builtin_registry, RegistryLoader
from numba.core import types
from numba.core.errors import NumbaNotImplementedError, NumbaTypeError
def create_overload_selector(self, kind):
    os = OverloadSelector()
    loader = RegistryLoader(builtin_registry)
    for impl, sig in loader.new_registrations(kind):
        os.append(impl, sig)
    return os