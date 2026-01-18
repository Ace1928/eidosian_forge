from ..language.parser import parse_value
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean, GraphQLEnumType,
from ..type.directives import DirectiveLocation, GraphQLDirective
from ..type.introspection import (TypeKind, __Directive, __DirectiveLocation,
from .value_from_ast import value_from_ast
def build_input_value_def_map(input_value_introspection, argument_type):
    return OrderedDict([(f['name'], build_input_value(f, argument_type)) for f in input_value_introspection])