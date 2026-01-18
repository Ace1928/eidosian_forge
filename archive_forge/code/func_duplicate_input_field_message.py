from ...error import GraphQLError
from .base import ValidationRule
@staticmethod
def duplicate_input_field_message(field_name):
    return 'There can only be one input field named "{}".'.format(field_name)