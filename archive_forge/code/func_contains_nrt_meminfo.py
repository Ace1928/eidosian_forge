from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
def contains_nrt_meminfo(self):
    """
        Recursively check all contained types for need for NRT meminfo.
        """
    return any((model.has_nrt_meminfo() for model in self.traverse_models()))