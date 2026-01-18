from typing import Any, Dict
from ...error import GraphQLError
from ...language import NameNode, FragmentDefinitionNode, VisitorAction, SKIP
from . import ASTValidationContext, ASTValidationRule
def enter_fragment_definition(self, node: FragmentDefinitionNode, *_args: Any) -> VisitorAction:
    known_fragment_names = self.known_fragment_names
    fragment_name = node.name.value
    if fragment_name in known_fragment_names:
        self.report_error(GraphQLError(f"There can be only one fragment named '{fragment_name}'.", [known_fragment_names[fragment_name], node.name]))
    else:
        known_fragment_names[fragment_name] = node.name
    return SKIP