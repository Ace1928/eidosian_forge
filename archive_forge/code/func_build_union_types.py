from collections import defaultdict
from typing import (
from ..language import (
from ..pyutils import inspect, merge_kwargs
from ..type import (
from .value_from_ast import value_from_ast
def build_union_types(nodes: Collection[Union[UnionTypeDefinitionNode, UnionTypeExtensionNode]]) -> List[GraphQLObjectType]:
    types: List[GraphQLObjectType] = []
    for node in nodes:
        for type_ in node.types or []:
            types.append(cast(GraphQLObjectType, get_named_type(type_)))
    return types