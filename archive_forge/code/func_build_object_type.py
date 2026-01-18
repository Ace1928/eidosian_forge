from collections import defaultdict
from ..error import GraphQLError
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type.definition import (GraphQLArgument, GraphQLEnumType,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..type.scalars import (GraphQLBoolean, GraphQLFloat, GraphQLID,
from ..type.schema import GraphQLSchema
from .value_from_ast import value_from_ast
def build_object_type(type_ast):
    return GraphQLObjectType(type_ast.name.value, interfaces=lambda: build_implemented_interfaces(type_ast), fields=lambda: build_field_map(type_ast))