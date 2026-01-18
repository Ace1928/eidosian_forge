from typing import Any, Callable, Dict, List, Optional, Union, cast
from ..language import print_ast, StringValueNode
from ..language.block_string import is_printable_as_block_string
from ..pyutils import inspect
from ..type import (
from .ast_from_value import ast_from_value
def print_type(type_: GraphQLNamedType) -> str:
    if is_scalar_type(type_):
        type_ = cast(GraphQLScalarType, type_)
        return print_scalar(type_)
    if is_object_type(type_):
        type_ = cast(GraphQLObjectType, type_)
        return print_object(type_)
    if is_interface_type(type_):
        type_ = cast(GraphQLInterfaceType, type_)
        return print_interface(type_)
    if is_union_type(type_):
        type_ = cast(GraphQLUnionType, type_)
        return print_union(type_)
    if is_enum_type(type_):
        type_ = cast(GraphQLEnumType, type_)
        return print_enum(type_)
    if is_input_object_type(type_):
        type_ = cast(GraphQLInputObjectType, type_)
        return print_input_object(type_)
    raise TypeError(f'Unexpected type: {inspect(type_)}.')