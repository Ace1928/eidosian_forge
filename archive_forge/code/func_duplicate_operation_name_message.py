from ...error import GraphQLError
from .base import ValidationRule
@staticmethod
def duplicate_operation_name_message(operation_name):
    return 'There can only be one operation named "{}".'.format(operation_name)