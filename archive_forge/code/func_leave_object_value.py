from typing import Any, Dict, List
from ...error import GraphQLError
from ...language import NameNode, ObjectFieldNode
from . import ASTValidationContext, ASTValidationRule
def leave_object_value(self, *_args: Any) -> None:
    self.known_names = self.known_names_stack.pop()