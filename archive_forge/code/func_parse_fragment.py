from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_fragment(parser):
    start = parser.token.start
    expect(parser, TokenKind.SPREAD)
    if peek(parser, TokenKind.NAME) and parser.token.value != 'on':
        return ast.FragmentSpread(name=parse_fragment_name(parser), directives=parse_directives(parser), loc=loc(parser, start))
    type_condition = None
    if parser.token.value == 'on':
        advance(parser)
        type_condition = parse_named_type(parser)
    return ast.InlineFragment(type_condition=type_condition, directives=parse_directives(parser), selection_set=parse_selection_set(parser), loc=loc(parser, start))