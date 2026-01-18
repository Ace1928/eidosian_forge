from ...error import GraphQLError
from ...language.printer import print_ast
from ...type.definition import is_input_type
from ...utils.type_from_ast import type_from_ast
from .base import ValidationRule
@staticmethod
def non_input_type_on_variable_message(variable_name, type_name):
    return 'Variable "${}" cannot be non-input type "{}".'.format(variable_name, type_name)