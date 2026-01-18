from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_variable(parser):
    start = parser.token.start
    expect(parser, TokenKind.DOLLAR)
    return ast.Variable(name=parse_name(parser), loc=loc(parser, start))