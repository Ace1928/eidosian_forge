from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_union_members(parser):
    members = []
    while True:
        members.append(parse_named_type(parser))
        if not skip(parser, TokenKind.PIPE):
            break
    return members