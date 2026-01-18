from .ast import (
def is_value_node(node: Node) -> bool:
    """Check whether the given node represents a value."""
    return isinstance(node, ValueNode)