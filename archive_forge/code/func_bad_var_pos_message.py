from ...error import GraphQLError
from ...type.definition import GraphQLNonNull
from ...utils.type_comparators import is_type_sub_type_of
from ...utils.type_from_ast import type_from_ast
from .base import ValidationRule
@staticmethod
def bad_var_pos_message(var_name, var_type, expected_type):
    return 'Variable "{}" of type "{}" used in position expecting type "{}".'.format(var_name, var_type, expected_type)