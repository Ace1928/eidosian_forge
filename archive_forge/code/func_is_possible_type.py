from collections.abc import Iterable
from .definition import GraphQLObjectType
from .directives import GraphQLDirective, specified_directives
from .introspection import IntrospectionSchema
from .typemap import GraphQLTypeMap
def is_possible_type(self, abstract_type, possible_type):
    return self._type_map.is_possible_type(abstract_type, possible_type)