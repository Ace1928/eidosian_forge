from ..language import ast
from ..language.printer import print_ast
from ..type.definition import (GraphQLEnumType, GraphQLInputObjectType,
def get_field_ast_value(field_name):
    if field_name in field_ast_map:
        return field_ast_map[field_name].value