from ..language.parser import parse_value
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean, GraphQLEnumType,
from ..type.directives import DirectiveLocation, GraphQLDirective
from ..type.introspection import (TypeKind, __Directive, __DirectiveLocation,
from .value_from_ast import value_from_ast
def build_interface_def(interface_introspection):
    return GraphQLInterfaceType(name=interface_introspection['name'], description=interface_introspection.get('description'), fields=lambda: build_field_def_map(interface_introspection), resolve_type=no_execution)