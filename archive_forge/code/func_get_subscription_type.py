from collections.abc import Iterable
from .definition import GraphQLObjectType
from .directives import GraphQLDirective, specified_directives
from .introspection import IntrospectionSchema
from .typemap import GraphQLTypeMap
def get_subscription_type(self):
    return self._subscription