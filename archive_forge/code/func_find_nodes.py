import io
from typing import Any, Dict, Generic, List, Tuple, Type, TypeVar, Union
import pickle  # noqa: F401
import ray
from ray.dag.base import DAGNodeBase
def find_nodes(self, obj: Any) -> List[SourceType]:
    """Find top-level DAGNodes."""
    assert self._found is None, 'find_nodes cannot be called twice on the same PyObjScanner instance.'
    self._found = []
    self._objects = []
    self.dump(obj)
    return self._found