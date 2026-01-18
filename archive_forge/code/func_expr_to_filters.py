import ast
import sys
from typing import Any
from .internal import Filters, Key
def expr_to_filters(expr: str) -> Filters:
    if not expr:
        filters = []
    else:
        parsed_expr = ast.parse(expr, mode='eval')
        filters = [_parse_node(parsed_expr.body)]
    return Filters(op='OR', filters=[Filters(op='AND', filters=filters)])