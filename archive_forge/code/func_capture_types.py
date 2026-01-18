import collections as py_collections
import functools
from typing import Any, Callable, Hashable, Mapping, Optional
from tensorflow.core.function import trace_type
from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util import object_identity
@property
def capture_types(self):
    if self._by_val_internal.mutated or self._by_val_external.mutated:
        self._recompute_cached_properties()
    return self._cached_capture_types