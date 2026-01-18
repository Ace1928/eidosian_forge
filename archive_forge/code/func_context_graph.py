import collections.abc
from typing import Any, Hashable, Optional, Dict
import weakref
from tensorflow.core.function.trace_type import default_types
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
@property
def context_graph(self):
    return self._context_graph