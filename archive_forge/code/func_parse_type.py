from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_type(parser):
    """Handles the 'Type': TypeName, ListType, and NonNullType
    parsing rules."""
    start = parser.token.start
    if skip(parser, TokenKind.BRACKET_L):
        ast_type = parse_type(parser)
        expect(parser, TokenKind.BRACKET_R)
        ast_type = ast.ListType(type=ast_type, loc=loc(parser, start))
    else:
        ast_type = parse_named_type(parser)
    if skip(parser, TokenKind.BANG):
        return ast.NonNullType(type=ast_type, loc=loc(parser, start))
    return ast_type