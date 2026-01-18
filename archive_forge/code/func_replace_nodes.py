import io
from typing import Any, Dict, Generic, List, Tuple, Type, TypeVar, Union
import pickle  # noqa: F401
import ray
from ray.dag.base import DAGNodeBase
def replace_nodes(self, table: Dict[SourceType, TransformedType]) -> Any:
    """Replace previously found DAGNodes per the given table."""
    assert self._found is not None, 'find_nodes must be called first'
    self._replace_table = table
    self._buf.seek(0)
    return pickle.load(self._buf)