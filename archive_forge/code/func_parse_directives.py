from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_directives(parser):
    directives = []
    while peek(parser, TokenKind.AT):
        directives.append(parse_directive(parser))
    return directives