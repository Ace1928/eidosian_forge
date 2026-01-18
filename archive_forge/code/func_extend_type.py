from collections import defaultdict
from ..error import GraphQLError
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type.definition import (GraphQLArgument, GraphQLEnumType,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..type.scalars import (GraphQLBoolean, GraphQLFloat, GraphQLID,
from ..type.schema import GraphQLSchema
from .value_from_ast import value_from_ast
def extend_type(type):
    if isinstance(type, GraphQLObjectType):
        return extend_object_type(type)
    if isinstance(type, GraphQLInterfaceType):
        return extend_interface_type(type)
    if isinstance(type, GraphQLUnionType):
        return extend_union_type(type)
    return type