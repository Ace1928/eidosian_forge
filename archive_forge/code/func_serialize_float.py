from math import isfinite
from typing import Any, Mapping
from ..error import GraphQLError
from ..pyutils import inspect
from ..language.ast import (
from ..language.printer import print_ast
from .definition import GraphQLNamedType, GraphQLScalarType
def serialize_float(output_value: Any) -> float:
    if isinstance(output_value, bool):
        return 1 if output_value else 0
    try:
        if not output_value and isinstance(output_value, str):
            output_value = ''
            raise ValueError
        num = output_value if isinstance(output_value, float) else float(output_value)
        if not isfinite(num):
            raise ValueError
    except (ValueError, TypeError):
        raise GraphQLError('Float cannot represent non numeric value: ' + inspect(output_value))
    return num