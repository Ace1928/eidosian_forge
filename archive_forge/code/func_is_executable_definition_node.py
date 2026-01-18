from .ast import (
def is_executable_definition_node(node: Node) -> bool:
    """Check whether the given node represents an executable definition."""
    return isinstance(node, ExecutableDefinitionNode)