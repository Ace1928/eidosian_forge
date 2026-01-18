from .ast import (
def is_type_node(node: Node) -> bool:
    """Check whether the given node represents a type."""
    return isinstance(node, TypeNode)