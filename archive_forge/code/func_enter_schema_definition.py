from typing import Any
from ...error import GraphQLError
from ...language import SchemaDefinitionNode
from . import SDLValidationRule, SDLValidationContext
def enter_schema_definition(self, node: SchemaDefinitionNode, *_args: Any) -> None:
    if self.already_defined:
        self.report_error(GraphQLError('Cannot define a new schema within a schema extension.', node))
    else:
        if self.schema_definitions_count:
            self.report_error(GraphQLError('Must provide only one schema definition.', node))
        self.schema_definitions_count += 1