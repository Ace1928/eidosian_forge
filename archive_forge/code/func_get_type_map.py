from collections.abc import Iterable
from .definition import GraphQLObjectType
from .directives import GraphQLDirective, specified_directives
from .introspection import IntrospectionSchema
from .typemap import GraphQLTypeMap
def get_type_map(self):
    return self._type_map