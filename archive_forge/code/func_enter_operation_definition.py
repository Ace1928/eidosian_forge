from typing import Any, Dict
from ...error import GraphQLError
from ...language import NameNode, FragmentDefinitionNode, VisitorAction, SKIP
from . import ASTValidationContext, ASTValidationRule
@staticmethod
def enter_operation_definition(*_args: Any) -> VisitorAction:
    return SKIP