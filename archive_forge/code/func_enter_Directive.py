from ...error import GraphQLError
from ...language import ast
from ...type.directives import DirectiveLocation
from .base import ValidationRule
def enter_Directive(self, node, key, parent, path, ancestors):
    directive_def = next((definition for definition in self.context.get_schema().get_directives() if definition.name == node.name.value), None)
    if not directive_def:
        return self.context.report_error(GraphQLError(self.unknown_directive_message(node.name.value), [node]))
    candidate_location = get_directive_location_for_ast_path(ancestors)
    if not candidate_location:
        self.context.report_error(GraphQLError(self.misplaced_directive_message(node.name.value, node.type), [node]))
    elif candidate_location not in directive_def.locations:
        self.context.report_error(GraphQLError(self.misplaced_directive_message(node.name.value, candidate_location), [node]))