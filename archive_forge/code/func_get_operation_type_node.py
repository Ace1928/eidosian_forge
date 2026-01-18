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
def get_operation_type_node(schema: GraphQLSchema, operation: OperationType) -> Optional[Node]:
    ast_node: Optional[Union[SchemaDefinitionNode, SchemaExtensionNode]]
    for ast_node in [schema.ast_node, *(schema.extension_ast_nodes or ())]:
        if ast_node:
            operation_types = ast_node.operation_types
            if operation_types:
                for operation_type in operation_types:
                    if operation_type.operation == operation:
                        return operation_type.type
    return None