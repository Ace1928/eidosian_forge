from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_fragment_name(parser):
    if parser.token.value == 'on':
        raise unexpected(parser)
    return parse_name(parser)