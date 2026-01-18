from ...error import GraphQLError
from .base import ValidationRule
def enter_FragmentDefinition(self, node, key, parent, path, ancestors):
    fragment_name = node.name.value
    if fragment_name in self.known_fragment_names:
        self.context.report_error(GraphQLError(self.duplicate_fragment_name_message(fragment_name), [self.known_fragment_names[fragment_name], node.name]))
    else:
        self.known_fragment_names[fragment_name] = node.name
    return False