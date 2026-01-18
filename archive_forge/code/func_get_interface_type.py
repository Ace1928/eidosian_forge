from ..language.parser import parse_value
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean, GraphQLEnumType,
from ..type.directives import DirectiveLocation, GraphQLDirective
from ..type.introspection import (TypeKind, __Directive, __DirectiveLocation,
from .value_from_ast import value_from_ast
def get_interface_type(type_ref):
    interface_type = get_type(type_ref)
    assert isinstance(interface_type, GraphQLInterfaceType), 'Introspection must provide interface type for interfaces.'
    return interface_type