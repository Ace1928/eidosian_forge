from ...error import GraphQLError
from ...type.definition import GraphQLNonNull
from ...utils.type_comparators import is_type_sub_type_of
from ...utils.type_from_ast import type_from_ast
from .base import ValidationRule
@staticmethod
def effective_type(var_type, var_def):
    if not var_def.default_value or isinstance(var_type, GraphQLNonNull):
        return var_type
    return GraphQLNonNull(var_type)