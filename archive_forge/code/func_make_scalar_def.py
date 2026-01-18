from ..execution.values import get_argument_values
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..utils.value_from_ast import value_from_ast
def make_scalar_def(definition):
    return GraphQLScalarType(name=definition.name.value, serialize=_none, parse_literal=_false, parse_value=_false)