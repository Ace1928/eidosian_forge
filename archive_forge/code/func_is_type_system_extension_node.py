from .ast import (
def is_type_system_extension_node(node: Node) -> bool:
    """Check whether the given node represents a type system extension."""
    return isinstance(node, (SchemaExtensionNode, TypeExtensionNode))