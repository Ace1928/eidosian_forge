from ...error import GraphQLError
from ...utils.quoted_or_list import quoted_or_list
from ...utils.suggestion_list import suggestion_list
from .base import ValidationRule
def enter_UnionTypeDefinition(self, node, *args):
    return False