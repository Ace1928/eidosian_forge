import collections as py_collections
import functools
from typing import Any, Callable, Hashable, Mapping, Optional
from tensorflow.core.function import trace_type
from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util import object_identity
def merge_by_ref_with(self, other: 'FunctionCaptures') -> None:
    """Add by-ref captures from `other` to `self` if not exist."""
    assert isinstance(other, FunctionCaptures)
    for key in other.by_ref_external:
        if key not in self._by_ref_external:
            self._by_ref_external[key] = other.by_ref_external[key]
            self._by_ref_tracetype[key] = other.by_ref_tracetype[key]