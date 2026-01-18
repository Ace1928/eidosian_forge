import collections as py_collections
import functools
from typing import Any, Callable, Hashable, Mapping, Optional
from tensorflow.core.function import trace_type
from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util import object_identity
def add_or_replace(self, key: Hashable, external: Any, internal: core.Tensor, tracetype: Any=None, is_by_ref: bool=False) -> None:
    """Replace a already exsiting capture, otherwise add it."""
    if is_by_ref:
        self._by_ref_external[key] = external
        self._by_ref_internal[key] = internal
        self._by_ref_tracetype[key] = tracetype
    else:
        self._by_val_internal[key] = internal
        self._by_val_external[key] = external
        if tracetype is not None:
            self._by_val_tracetype[key] = tracetype
        else:
            self._by_val_tracetype[key] = trace_type.from_value(external)