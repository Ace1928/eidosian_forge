from collections import Counter
from ...error import GraphQLError
from ...pyutils.ordereddict import OrderedDict
from ...type.definition import (GraphQLInterfaceType, GraphQLObjectType,
from ...utils.quoted_or_list import quoted_or_list
from ...utils.suggestion_list import suggestion_list
from .base import ValidationRule
def get_suggested_field_names(schema, graphql_type, field_name):
    """For the field name provided, determine if there are any similar field names
    that may be the result of a typo."""
    if isinstance(graphql_type, (GraphQLInterfaceType, GraphQLObjectType)):
        possible_field_names = list(graphql_type.fields.keys())
        return suggestion_list(field_name, possible_field_names)
    return []