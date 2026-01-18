from typing import cast, Any
from ...error import GraphQLError
from ...language import (
from ...pyutils import did_you_mean, suggestion_list, Undefined
from ...type import (
from . import ValidationRule
def is_valid_value_node(self, node: ValueNode) -> None:
    """Check whether this is a valid value node.

        Any value literal may be a valid representation of a Scalar, depending on that
        scalar type.
        """
    location_type = self.context.get_input_type()
    if not location_type:
        return
    type_ = get_named_type(location_type)
    if not is_leaf_type(type_):
        self.report_error(GraphQLError(f"Expected value of type '{location_type}', found {print_ast(node)}.", node))
        return
    type_ = cast(GraphQLScalarType, type_)
    try:
        parse_result = type_.parse_literal(node)
        if parse_result is Undefined:
            self.report_error(GraphQLError(f"Expected value of type '{location_type}', found {print_ast(node)}.", node))
    except GraphQLError as error:
        self.report_error(error)
    except Exception as error:
        self.report_error(GraphQLError(f"Expected value of type '{location_type}', found {print_ast(node)}; {error}", node, original_error=error))
    return