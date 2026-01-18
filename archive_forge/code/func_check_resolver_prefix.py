from .error import *
from .nodes import *
import re
def check_resolver_prefix(self, depth, path, kind, current_node, current_index):
    node_check, index_check = path[depth - 1]
    if isinstance(node_check, str):
        if current_node.tag != node_check:
            return
    elif node_check is not None:
        if not isinstance(current_node, node_check):
            return
    if index_check is True and current_index is not None:
        return
    if (index_check is False or index_check is None) and current_index is None:
        return
    if isinstance(index_check, str):
        if not (isinstance(current_index, ScalarNode) and index_check == current_index.value):
            return
    elif isinstance(index_check, int) and (not isinstance(index_check, bool)):
        if index_check != current_index:
            return
    return True