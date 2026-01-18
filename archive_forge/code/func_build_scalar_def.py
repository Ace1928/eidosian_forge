from ..language.parser import parse_value
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean, GraphQLEnumType,
from ..type.directives import DirectiveLocation, GraphQLDirective
from ..type.introspection import (TypeKind, __Directive, __DirectiveLocation,
from .value_from_ast import value_from_ast
def build_scalar_def(scalar_introspection):
    return GraphQLScalarType(name=scalar_introspection['name'], description=scalar_introspection.get('description'), serialize=_none, parse_value=_false, parse_literal=_false)