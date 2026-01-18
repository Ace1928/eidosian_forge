from typing import TYPE_CHECKING
from types import SimpleNamespace
@property
def current_node_id(self) -> 'NodeID':
    from ray import NodeID
    return NodeID(self._fetch_runtime_context().node_id)