from copy import copy
from typing import Tuple
from ..language import ListValueNode, ObjectFieldNode, ObjectValueNode, ValueNode
from ..pyutils import natural_comparison_key
def sort_value_node(value_node: ValueNode) -> ValueNode:
    """Sort ValueNode.

    This function returns a sorted copy of the given ValueNode

    For internal use only.
    """
    if isinstance(value_node, ObjectValueNode):
        value_node = copy(value_node)
        value_node.fields = sort_fields(value_node.fields)
    elif isinstance(value_node, ListValueNode):
        value_node = copy(value_node)
        value_node.values = tuple((sort_value_node(value) for value in value_node.values))
    return value_node