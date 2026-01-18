import collections as py_collections
import functools
from typing import Any, Callable, Hashable, Mapping, Optional
from tensorflow.core.function import trace_type
from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util import object_identity
def get_by_ref_snapshot(self) -> Mapping[Hashable, Any]:
    """Get a snapshot of current values of by-ref captures."""
    snapshot = {}
    for key in self._by_ref_external:
        func = self._by_ref_external[key]
        try:
            value = func()
        except (AttributeError, RuntimeError):
            value = self._by_ref_tracetype[key]
        snapshot[key] = value
    return snapshot