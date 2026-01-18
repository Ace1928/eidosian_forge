from .ast import (
def is_type_definition_node(node: Node) -> bool:
    """Check whether the given node represents a type definition."""
    return isinstance(node, TypeDefinitionNode)