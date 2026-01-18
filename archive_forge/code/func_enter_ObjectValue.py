from ...error import GraphQLError
from .base import ValidationRule
def enter_ObjectValue(self, node, key, parent, path, ancestors):
    self.known_names_stack.append(self.known_names)
    self.known_names = {}