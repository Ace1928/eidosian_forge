from ..language.printer import print_ast
from ..type.definition import (GraphQLEnumType, GraphQLInputObjectType,
from ..type.directives import DEFAULT_DEPRECATION_REASON
from .ast_from_value import ast_from_value
def print_schema(schema):
    return _print_filtered_schema(schema, lambda n: not is_spec_directive(n), _is_defined_type)