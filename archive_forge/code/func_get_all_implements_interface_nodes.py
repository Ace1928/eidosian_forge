from operator import attrgetter, itemgetter
from typing import (
from ..error import GraphQLError
from ..pyutils import inspect
from ..language import (
from .definition import (
from ..utilities.type_comparators import is_equal_type, is_type_sub_type_of
from .directives import is_directive, GraphQLDeprecatedDirective
from .introspection import is_introspection_type
from .schema import GraphQLSchema, assert_schema
def get_all_implements_interface_nodes(type_: Union[GraphQLObjectType, GraphQLInterfaceType], iface: GraphQLInterfaceType) -> List[NamedTypeNode]:
    ast_node = type_.ast_node
    nodes = type_.extension_ast_nodes
    if ast_node is not None:
        nodes = [ast_node, *nodes]
    implements_nodes: List[NamedTypeNode] = []
    for node in nodes:
        iface_nodes = node.interfaces
        if iface_nodes:
            implements_nodes.extend((iface_node for iface_node in iface_nodes if iface_node.name.value == iface.name))
    return implements_nodes