from collections.abc import Iterable
import decimal
from functools import partial
from wandb_graphql.language import ast
from wandb_graphql.language.printer import print_ast
from wandb_graphql.type import (GraphQLField, GraphQLList,
from .utils import to_camel_case
def get_arg_serializer(arg_type):
    if isinstance(arg_type, GraphQLNonNull):
        return get_arg_serializer(arg_type.of_type)
    if isinstance(arg_type, GraphQLList):
        inner_serializer = get_arg_serializer(arg_type.of_type)
        return partial(serialize_list, inner_serializer)
    if isinstance(arg_type, GraphQLEnumType):
        return lambda value: ast.EnumValue(value=arg_type.serialize(value))
    return arg_type.serialize