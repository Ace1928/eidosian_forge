from ..language.ast import BooleanValue, FloatValue, IntValue, StringValue
from .definition import GraphQLScalarType
def parse_string_literal(ast):
    if isinstance(ast, StringValue):
        return ast.value
    return None