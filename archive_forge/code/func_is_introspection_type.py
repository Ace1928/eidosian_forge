from enum import Enum
from typing import Mapping
from .definition import (
from ..language import DirectiveLocation, print_ast
from ..pyutils import inspect
from .scalars import GraphQLBoolean, GraphQLString
def is_introspection_type(type_: GraphQLNamedType) -> bool:
    """Check whether the given named GraphQL type is an introspection type."""
    return type_.name in introspection_types