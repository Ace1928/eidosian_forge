from ...error import GraphQLError
from .base import ValidationRule
def leave_ObjectValue(self, node, key, parent, path, ancestors):
    self.known_names = self.known_names_stack.pop()