from ...error import GraphQLError
from .base import ValidationRule
def enter_Argument(self, node, key, parent, path, ancestors):
    arg_name = node.name.value
    if arg_name in self.known_arg_names:
        self.context.report_error(GraphQLError(self.duplicate_arg_message(arg_name), [self.known_arg_names[arg_name], node.name]))
    else:
        self.known_arg_names[arg_name] = node.name
    return False