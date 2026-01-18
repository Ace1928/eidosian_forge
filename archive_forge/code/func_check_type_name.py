from typing import Any, Dict
from ...error import GraphQLError
from ...language import NameNode, TypeDefinitionNode, VisitorAction, SKIP
from . import SDLValidationContext, SDLValidationRule
def check_type_name(self, node: TypeDefinitionNode, *_args: Any) -> VisitorAction:
    type_name = node.name.value
    if self.schema and self.schema.get_type(type_name):
        self.report_error(GraphQLError(f"Type '{type_name}' already exists in the schema. It cannot also be defined in this type definition.", node.name))
    else:
        if type_name in self.known_type_names:
            self.report_error(GraphQLError(f"There can be only one type named '{type_name}'.", [self.known_type_names[type_name], node.name]))
        else:
            self.known_type_names[type_name] = node.name
        return SKIP
    return None