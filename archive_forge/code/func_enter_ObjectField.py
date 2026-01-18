from ...error import GraphQLError
from .base import ValidationRule
def enter_ObjectField(self, node, key, parent, path, ancestors):
    field_name = node.name.value
    if field_name in self.known_names:
        self.context.report_error(GraphQLError(self.duplicate_input_field_message(field_name), [self.known_names[field_name], node.name]))
    else:
        self.known_names[field_name] = node.name
    return False