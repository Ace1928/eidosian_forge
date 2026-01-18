from .ast import (
def is_const_value_node(node: Node) -> bool:
    """Check whether the given node represents a constant value."""
    return is_value_node(node) and (any((is_const_value_node(value) for value in node.values)) if isinstance(node, ListValueNode) else any((is_const_value_node(field.value) for field in node.fields)) if isinstance(node, ObjectValueNode) else not isinstance(node, VariableNode))