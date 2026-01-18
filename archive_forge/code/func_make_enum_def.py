from ..execution.values import get_argument_values
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..utils.value_from_ast import value_from_ast
def make_enum_def(definition):
    values = OrderedDict(((v.name.value, GraphQLEnumValue(deprecation_reason=get_deprecation_reason(v.directives))) for v in definition.values))
    return GraphQLEnumType(name=definition.name.value, values=values)