from collections import defaultdict
from ..error import GraphQLError
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type.definition import (GraphQLArgument, GraphQLEnumType,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..type.scalars import (GraphQLBoolean, GraphQLFloat, GraphQLID,
from ..type.schema import GraphQLSchema
from .value_from_ast import value_from_ast
def extend_field_type(type):
    if isinstance(type, GraphQLList):
        return GraphQLList(extend_field_type(type.of_type))
    if isinstance(type, GraphQLNonNull):
        return GraphQLNonNull(extend_field_type(type.of_type))
    return get_type_from_def(type)