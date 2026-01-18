from ..language.parser import parse_value
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean, GraphQLEnumType,
from ..type.directives import DirectiveLocation, GraphQLDirective
from ..type.introspection import (TypeKind, __Directive, __DirectiveLocation,
from .value_from_ast import value_from_ast
def build_object_def(object_introspection):
    return GraphQLObjectType(name=object_introspection['name'], description=object_introspection.get('description'), interfaces=[get_interface_type(i) for i in object_introspection.get('interfaces', [])], fields=lambda: build_field_def_map(object_introspection))