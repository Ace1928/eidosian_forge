from copy import copy, deepcopy
from typing import (
from ..error import GraphQLError
from ..language import ast, OperationType
from ..pyutils import inspect, is_collection, is_description
from .definition import (
from .directives import GraphQLDirective, specified_directives, is_directive
from .introspection import introspection_types
def is_sub_type(self, abstract_type: GraphQLAbstractType, maybe_sub_type: GraphQLNamedType) -> bool:
    """Check whether a type is a subtype of a given abstract type."""
    types = self._sub_type_map.get(abstract_type.name)
    if types is None:
        types = set()
        add = types.add
        if is_union_type(abstract_type):
            for type_ in cast(GraphQLUnionType, abstract_type).types:
                add(type_.name)
        else:
            implementations = self.get_implementations(cast(GraphQLInterfaceType, abstract_type))
            for type_ in implementations.objects:
                add(type_.name)
            for type_ in implementations.interfaces:
                add(type_.name)
        self._sub_type_map[abstract_type.name] = types
    return maybe_sub_type.name in types