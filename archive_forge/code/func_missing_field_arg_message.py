from ...error import GraphQLError
from ...type.definition import GraphQLNonNull
from .base import ValidationRule
@staticmethod
def missing_field_arg_message(name, arg_name, type):
    return 'Field "{}" argument "{}" of type "{}" is required but not provided.'.format(name, arg_name, type)