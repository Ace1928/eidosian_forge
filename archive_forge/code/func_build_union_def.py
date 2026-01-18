from ..language.parser import parse_value
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean, GraphQLEnumType,
from ..type.directives import DirectiveLocation, GraphQLDirective
from ..type.introspection import (TypeKind, __Directive, __DirectiveLocation,
from .value_from_ast import value_from_ast
def build_union_def(union_introspection):
    return GraphQLUnionType(name=union_introspection['name'], description=union_introspection.get('description'), types=[get_object_type(t) for t in union_introspection.get('possibleTypes', [])], resolve_type=no_execution)