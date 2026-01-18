from .ast import (
def is_selection_node(node: Node) -> bool:
    """Check whether the given node represents a selection."""
    return isinstance(node, SelectionNode)