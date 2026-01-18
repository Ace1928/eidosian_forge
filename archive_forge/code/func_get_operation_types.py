from collections import defaultdict
from typing import (
from ..language import (
from ..pyutils import inspect, merge_kwargs
from ..type import (
from .value_from_ast import value_from_ast
def get_operation_types(nodes: Collection[Union[SchemaDefinitionNode, SchemaExtensionNode]]) -> Dict[OperationType, GraphQLNamedType]:
    return {operation_type.operation: get_named_type(operation_type.type) for node in nodes for operation_type in node.operation_types or []}