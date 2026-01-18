from typing import Any, Dict, Optional, Union
from ...error import GraphQLError
from ...language import (
from ...type import GraphQLObjectType
from . import SDLValidationContext, SDLValidationRule
def check_operation_types(self, node: Union[SchemaDefinitionNode, SchemaExtensionNode], *_args: Any) -> VisitorAction:
    for operation_type in node.operation_types or []:
        operation = operation_type.operation
        already_defined_operation_type = self.defined_operation_types.get(operation)
        if self.existing_operation_types.get(operation):
            self.report_error(GraphQLError(f'Type for {operation.value} already defined in the schema. It cannot be redefined.', operation_type))
        elif already_defined_operation_type:
            self.report_error(GraphQLError(f'There can be only one {operation.value} type in schema.', [already_defined_operation_type, operation_type]))
        else:
            self.defined_operation_types[operation] = operation_type
    return SKIP