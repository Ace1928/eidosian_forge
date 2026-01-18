from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_selection(parser):
    if peek(parser, TokenKind.SPREAD):
        return parse_fragment(parser)
    else:
        return parse_field(parser)