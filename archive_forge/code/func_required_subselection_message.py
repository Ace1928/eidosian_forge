from ...error import GraphQLError
from ...type.definition import get_named_type, is_leaf_type
from .base import ValidationRule
@staticmethod
def required_subselection_message(field, type):
    return 'Field "{}" of type "{}" must have a sub selection.'.format(field, type)