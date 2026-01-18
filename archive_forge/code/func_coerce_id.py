from math import isfinite
from typing import Any, Mapping
from ..error import GraphQLError
from ..pyutils import inspect
from ..language.ast import (
from ..language.printer import print_ast
from .definition import GraphQLNamedType, GraphQLScalarType
def coerce_id(input_value: Any) -> str:
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, int) and (not isinstance(input_value, bool)):
        return str(input_value)
    if isinstance(input_value, float) and isfinite(input_value) and (int(input_value) == input_value):
        return str(int(input_value))
    raise GraphQLError('ID cannot represent value: ' + inspect(input_value))