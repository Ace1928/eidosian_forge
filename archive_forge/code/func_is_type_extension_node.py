from .ast import (
def is_type_extension_node(node: Node) -> bool:
    """Check whether the given node represents a type extension."""
    return isinstance(node, TypeExtensionNode)