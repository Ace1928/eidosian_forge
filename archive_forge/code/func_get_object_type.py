from ..execution.values import get_argument_values
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..utils.value_from_ast import value_from_ast
def get_object_type(type_ast):
    type = type_def_named(type_ast.name.value)
    assert isinstance(type, GraphQLObjectType), 'AST must provide object type'
    return type