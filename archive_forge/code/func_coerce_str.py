from ..language.ast import BooleanValue, FloatValue, IntValue, StringValue
from .definition import GraphQLScalarType
def coerce_str(value):
    if isinstance(value, str):
        return value
    return str(value)