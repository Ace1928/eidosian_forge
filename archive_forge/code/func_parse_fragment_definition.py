from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_fragment_definition(parser):
    start = parser.token.start
    expect_keyword(parser, 'fragment')
    return ast.FragmentDefinition(name=parse_fragment_name(parser), type_condition=expect_keyword(parser, 'on') and parse_named_type(parser), directives=parse_directives(parser), selection_set=parse_selection_set(parser), loc=loc(parser, start))