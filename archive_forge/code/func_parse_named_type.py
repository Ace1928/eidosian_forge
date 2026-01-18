from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_named_type(parser):
    start = parser.token.start
    return ast.NamedType(name=parse_name(parser), loc=loc(parser, start))