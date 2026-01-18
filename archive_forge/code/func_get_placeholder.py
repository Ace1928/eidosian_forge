import collections.abc
from typing import Any, Hashable, Optional, Dict
import weakref
from tensorflow.core.function.trace_type import default_types
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
def get_placeholder(self, alias_id: Hashable) -> Hashable:
    if not self.has_placeholder(alias_id):
        raise KeyError(f'alias_id: {alias_id} not found in this instance of placeholder context.')
    return self._alias_id_to_placeholder[alias_id]