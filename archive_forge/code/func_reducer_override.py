import io
from typing import Any, Dict, Generic, List, Tuple, Type, TypeVar, Union
import pickle  # noqa: F401
import ray
from ray.dag.base import DAGNodeBase
def reducer_override(self, obj):
    """Hook for reducing objects.

        Objects of `self.source_type` are saved to `self._found` and a global map so
        they can later be replaced.

        All other objects fall back to the default `CloudPickler` serialization.
        """
    if isinstance(obj, self.source_type):
        index = len(self._found)
        self._found.append(obj)
        return (_get_node, (id(self), index))
    return super().reducer_override(obj)