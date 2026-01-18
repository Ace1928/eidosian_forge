from copy import copy, deepcopy
from typing import (
from ..error import GraphQLError
from ..language import ast, OperationType
from ..pyutils import inspect, is_collection, is_description
from .definition import (
from .directives import GraphQLDirective, specified_directives, is_directive
from .introspection import introspection_types
def assert_schema(schema: Any) -> GraphQLSchema:
    if not is_schema(schema):
        raise TypeError(f'Expected {inspect(schema)} to be a GraphQL schema.')
    return cast(GraphQLSchema, schema)