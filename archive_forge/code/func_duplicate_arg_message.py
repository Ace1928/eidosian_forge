from ...error import GraphQLError
from .base import ValidationRule
@staticmethod
def duplicate_arg_message(field):
    return 'There can only be one argument named "{}".'.format(field)