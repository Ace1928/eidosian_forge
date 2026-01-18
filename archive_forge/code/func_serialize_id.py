from math import isfinite
from typing import Any, Mapping
from ..error import GraphQLError
from ..pyutils import inspect
from ..language.ast import (
from ..language.printer import print_ast
from .definition import GraphQLNamedType, GraphQLScalarType
def serialize_id(output_value: Any) -> str:
    if isinstance(output_value, str):
        return output_value
    if isinstance(output_value, int) and (not isinstance(output_value, bool)):
        return str(output_value)
    if isinstance(output_value, float) and isfinite(output_value) and (int(output_value) == output_value):
        return str(int(output_value))
    if type(output_value).__module__ == 'builtins':
        raise GraphQLError('ID cannot represent value: ' + inspect(output_value))
    return str(output_value)