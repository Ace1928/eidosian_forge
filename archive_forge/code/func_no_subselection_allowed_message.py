from ...error import GraphQLError
from ...type.definition import get_named_type, is_leaf_type
from .base import ValidationRule
@staticmethod
def no_subselection_allowed_message(field, type):
    return 'Field "{}" of type "{}" must not have a sub selection.'.format(field, type)