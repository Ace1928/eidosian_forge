from collections import defaultdict
from typing import (
from ..language import (
from ..pyutils import inspect, merge_kwargs
from ..type import (
from .value_from_ast import value_from_ast
def extend_named_type(type_: GraphQLNamedType) -> GraphQLNamedType:
    if is_introspection_type(type_) or is_specified_scalar_type(type_):
        return type_
    if is_scalar_type(type_):
        type_ = cast(GraphQLScalarType, type_)
        return extend_scalar_type(type_)
    if is_object_type(type_):
        type_ = cast(GraphQLObjectType, type_)
        return extend_object_type(type_)
    if is_interface_type(type_):
        type_ = cast(GraphQLInterfaceType, type_)
        return extend_interface_type(type_)
    if is_union_type(type_):
        type_ = cast(GraphQLUnionType, type_)
        return extend_union_type(type_)
    if is_enum_type(type_):
        type_ = cast(GraphQLEnumType, type_)
        return extend_enum_type(type_)
    if is_input_object_type(type_):
        type_ = cast(GraphQLInputObjectType, type_)
        return extend_input_object_type(type_)
    raise TypeError(f'Unexpected type: {inspect(type_)}.')