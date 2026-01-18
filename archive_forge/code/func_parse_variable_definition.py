from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_variable_definition(parser):
    start = parser.token.start
    return ast.VariableDefinition(variable=parse_variable(parser), type=expect(parser, TokenKind.COLON) and parse_type(parser), default_value=parse_value_literal(parser, True) if skip(parser, TokenKind.EQUALS) else None, loc=loc(parser, start))