from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_name(parser):
    """Converts a name lex token into a name parse node."""
    token = expect(parser, TokenKind.NAME)
    return ast.Name(value=token.value, loc=loc(parser, token.start))