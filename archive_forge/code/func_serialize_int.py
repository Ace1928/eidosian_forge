from math import isfinite
from typing import Any, Mapping
from ..error import GraphQLError
from ..pyutils import inspect
from ..language.ast import (
from ..language.printer import print_ast
from .definition import GraphQLNamedType, GraphQLScalarType
def serialize_int(output_value: Any) -> int:
    if isinstance(output_value, bool):
        return 1 if output_value else 0
    try:
        if isinstance(output_value, int):
            num = output_value
        elif isinstance(output_value, float):
            num = int(output_value)
            if num != output_value:
                raise ValueError
        elif not output_value and isinstance(output_value, str):
            output_value = ''
            raise ValueError
        else:
            num = int(output_value)
    except (OverflowError, ValueError, TypeError):
        raise GraphQLError('Int cannot represent non-integer value: ' + inspect(output_value))
    if not GRAPHQL_MIN_INT <= num <= GRAPHQL_MAX_INT:
        raise GraphQLError('Int cannot represent non 32-bit signed integer value: ' + inspect(output_value))
    return num