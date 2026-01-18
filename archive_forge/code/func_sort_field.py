from copy import copy
from typing import Tuple
from ..language import ListValueNode, ObjectFieldNode, ObjectValueNode, ValueNode
from ..pyutils import natural_comparison_key
def sort_field(field: ObjectFieldNode) -> ObjectFieldNode:
    field = copy(field)
    field.value = sort_value_node(field.value)
    return field