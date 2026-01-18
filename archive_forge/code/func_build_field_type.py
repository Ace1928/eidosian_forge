from collections import defaultdict
from ..error import GraphQLError
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type.definition import (GraphQLArgument, GraphQLEnumType,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..type.scalars import (GraphQLBoolean, GraphQLFloat, GraphQLID,
from ..type.schema import GraphQLSchema
from .value_from_ast import value_from_ast
def build_field_type(type_ast):
    if isinstance(type_ast, ast.ListType):
        return GraphQLList(build_field_type(type_ast.type))
    if isinstance(type_ast, ast.NonNullType):
        return GraphQLNonNull(build_field_type(type_ast.type))
    return get_type_from_AST(type_ast)