from enum import Enum
from typing import (
from ..error import GraphQLError
from ..language import (
from ..pyutils import (
from ..utilities.value_from_ast_untyped import value_from_ast_untyped
from .assert_name import assert_name, assert_enum_value_name
def assert_list_type(type_: Any) -> GraphQLList:
    if not is_list_type(type_):
        raise TypeError(f'Expected {type_} to be a GraphQL List type.')
    return cast(GraphQLList, type_)