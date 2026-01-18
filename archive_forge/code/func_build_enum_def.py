from ..language.parser import parse_value
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean, GraphQLEnumType,
from ..type.directives import DirectiveLocation, GraphQLDirective
from ..type.introspection import (TypeKind, __Directive, __DirectiveLocation,
from .value_from_ast import value_from_ast
def build_enum_def(enum_introspection):
    return GraphQLEnumType(name=enum_introspection['name'], description=enum_introspection.get('description'), values=OrderedDict([(value_introspection['name'], GraphQLEnumValue(description=value_introspection.get('description'), deprecation_reason=value_introspection.get('deprecationReason'))) for value_introspection in enum_introspection.get('enumValues', [])]))