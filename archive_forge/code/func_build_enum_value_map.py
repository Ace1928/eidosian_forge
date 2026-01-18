from collections import defaultdict
from typing import (
from ..language import (
from ..pyutils import inspect, merge_kwargs
from ..type import (
from .value_from_ast import value_from_ast
def build_enum_value_map(nodes: Collection[Union[EnumTypeDefinitionNode, EnumTypeExtensionNode]]) -> GraphQLEnumValueMap:
    enum_value_map: GraphQLEnumValueMap = {}
    for node in nodes:
        for value in node.values or []:
            value_name = value.name.value
            enum_value_map[value_name] = GraphQLEnumValue(value=value_name, description=value.description.value if value.description else None, deprecation_reason=get_deprecation_reason(value), ast_node=value)
    return enum_value_map