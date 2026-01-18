from ..language.ast import BooleanValue, FloatValue, IntValue, StringValue
from .definition import GraphQLScalarType
def coerce_int(value):
    if isinstance(value, int):
        num = value
    else:
        try:
            num = int(value)
        except ValueError:
            num = int(float(value))
    if MIN_INT <= num <= MAX_INT:
        return num
    raise Exception('Int cannot represent non 32-bit signed integer value: {}'.format(value))