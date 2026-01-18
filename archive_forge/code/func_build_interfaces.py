from collections import defaultdict
from typing import (
from ..language import (
from ..pyutils import inspect, merge_kwargs
from ..type import (
from .value_from_ast import value_from_ast
def build_interfaces(nodes: Collection[Union[InterfaceTypeDefinitionNode, InterfaceTypeExtensionNode, ObjectTypeDefinitionNode, ObjectTypeExtensionNode]]) -> List[GraphQLInterfaceType]:
    interfaces: List[GraphQLInterfaceType] = []
    for node in nodes:
        for type_ in node.interfaces or []:
            interfaces.append(cast(GraphQLInterfaceType, get_named_type(type_)))
    return interfaces