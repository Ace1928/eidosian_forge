import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def process_common_prefix_nodes(self_node, self_path, basis_node, basis_path):
    self_node = self._get_node(self_node)
    basis_node = basis._get_node(basis_node)
    if isinstance(self_node, InternalNode) and isinstance(basis_node, InternalNode):
        process_common_internal_nodes(self_node, basis_node)
    elif isinstance(self_node, LeafNode) and isinstance(basis_node, LeafNode):
        process_common_leaf_nodes(self_node, basis_node)
    else:
        process_node(self_node, self_path, self, self_pending)
        process_node(basis_node, basis_path, basis, basis_pending)