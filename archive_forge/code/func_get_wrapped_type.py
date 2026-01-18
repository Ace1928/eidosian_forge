from collections import defaultdict
from typing import (
from ..language import (
from ..pyutils import inspect, merge_kwargs
from ..type import (
from .value_from_ast import value_from_ast
def get_wrapped_type(node: TypeNode) -> GraphQLType:
    if isinstance(node, ListTypeNode):
        return GraphQLList(get_wrapped_type(node.type))
    if isinstance(node, NonNullTypeNode):
        return GraphQLNonNull(cast(GraphQLNullableType, get_wrapped_type(node.type)))
    return get_named_type(cast(NamedTypeNode, node))