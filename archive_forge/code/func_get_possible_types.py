from collections.abc import Iterable
from .definition import GraphQLObjectType
from .directives import GraphQLDirective, specified_directives
from .introspection import IntrospectionSchema
from .typemap import GraphQLTypeMap
def get_possible_types(self, abstract_type):
    return self._type_map.get_possible_types(abstract_type)