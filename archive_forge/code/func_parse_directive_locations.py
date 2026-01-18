from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_directive_locations(parser):
    locations = []
    while True:
        locations.append(parse_name(parser))
        if not skip(parser, TokenKind.PIPE):
            break
    return locations