from collections.abc import Iterable
from .definition import GraphQLObjectType
from .directives import GraphQLDirective, specified_directives
from .introspection import IntrospectionSchema
from .typemap import GraphQLTypeMap
def get_directives(self):
    return self._directives