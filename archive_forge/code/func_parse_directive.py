from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_directive(parser):
    start = parser.token.start
    expect(parser, TokenKind.AT)
    return ast.Directive(name=parse_name(parser), arguments=parse_arguments(parser), loc=loc(parser, start))