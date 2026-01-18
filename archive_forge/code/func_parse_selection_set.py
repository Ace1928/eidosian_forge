from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_selection_set(parser):
    start = parser.token.start
    return ast.SelectionSet(selections=many(parser, TokenKind.BRACE_L, parse_selection, TokenKind.BRACE_R), loc=loc(parser, start))