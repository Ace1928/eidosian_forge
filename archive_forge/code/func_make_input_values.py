from ..execution.values import get_argument_values
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..utils.value_from_ast import value_from_ast
def make_input_values(values, cls):
    return OrderedDict(((value.name.value, cls(type=produce_type_def(value.type), default_value=value_from_ast(value.default_value, produce_type_def(value.type)))) for value in values))