from ...error import GraphQLError
from .base import ValidationRule
@staticmethod
def duplicate_variable_message(operation_name):
    return 'There can be only one variable named "{}".'.format(operation_name)