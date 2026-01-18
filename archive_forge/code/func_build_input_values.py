from collections import defaultdict
from ..error import GraphQLError
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type.definition import (GraphQLArgument, GraphQLEnumType,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..type.scalars import (GraphQLBoolean, GraphQLFloat, GraphQLID,
from ..type.schema import GraphQLSchema
from .value_from_ast import value_from_ast
def build_input_values(values, input_type=GraphQLArgument):
    input_values = OrderedDict()
    for value in values:
        type = build_field_type(value.type)
        input_values[value.name.value] = input_type(type, default_value=value_from_ast(value.default_value, type))
    return input_values