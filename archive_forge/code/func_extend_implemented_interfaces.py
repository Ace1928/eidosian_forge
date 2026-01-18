from collections import defaultdict
from ..error import GraphQLError
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type.definition import (GraphQLArgument, GraphQLEnumType,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..type.scalars import (GraphQLBoolean, GraphQLFloat, GraphQLID,
from ..type.schema import GraphQLSchema
from .value_from_ast import value_from_ast
def extend_implemented_interfaces(type):
    interfaces = list(map(get_type_from_def, type.interfaces))
    extensions = type_extensions_map[type.name]
    for extension in extensions:
        for namedType in extension.definition.interfaces:
            interface_name = namedType.name.value
            if any([_def.name == interface_name for _def in interfaces]):
                raise GraphQLError(('Type "{}" already implements "{}". ' + 'It cannot also be implemented in this type extension.').format(type.name, interface_name), [namedType])
            interfaces.append(get_type_from_AST(namedType))
    return interfaces