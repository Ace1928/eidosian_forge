from ...error import GraphQLError
from ...language import ast
from .base import ValidationRule
def enter_Document(self, node, key, parent, path, ancestors):
    self.operation_count = sum((1 for definition in node.definitions if isinstance(definition, ast.OperationDefinition)))