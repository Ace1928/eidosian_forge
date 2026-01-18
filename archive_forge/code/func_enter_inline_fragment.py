from typing import Any
from ...error import GraphQLError
from ...language import (
from ...type import is_composite_type
from ...utilities import type_from_ast
from . import ValidationRule
def enter_inline_fragment(self, node: InlineFragmentNode, *_args: Any) -> None:
    type_condition = node.type_condition
    if type_condition:
        type_ = type_from_ast(self.context.schema, type_condition)
        if type_ and (not is_composite_type(type_)):
            type_str = print_ast(type_condition)
            self.report_error(GraphQLError(f"Fragment cannot condition on non composite type '{type_str}'.", type_condition))