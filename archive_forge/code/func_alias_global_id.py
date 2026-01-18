import collections.abc
from typing import Any, Hashable, Optional, Dict
import weakref
from tensorflow.core.function.trace_type import default_types
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
def alias_global_id(self, global_id: Hashable) -> Hashable:
    if global_id not in self._global_to_local_id:
        self._global_to_local_id[global_id] = len(self._global_to_local_id)
    return self._global_to_local_id[global_id]