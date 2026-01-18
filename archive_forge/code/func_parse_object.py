from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_object(parser, is_const):
    start = parser.token.start
    expect(parser, TokenKind.BRACE_L)
    fields = []
    while not skip(parser, TokenKind.BRACE_R):
        fields.append(parse_object_field(parser, is_const))
    return ast.ObjectValue(fields=fields, loc=loc(parser, start))