from typing import cast, Any
from ...error import GraphQLError
from ...language import (
from ...pyutils import did_you_mean, suggestion_list, Undefined
from ...type import (
from . import ValidationRule
def enter_null_value(self, node: NullValueNode, *_args: Any) -> None:
    type_ = self.context.get_input_type()
    if is_non_null_type(type_):
        self.report_error(GraphQLError(f"Expected value of type '{type_}', found {print_ast(node)}.", node))