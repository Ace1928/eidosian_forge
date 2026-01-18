from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_union_type_definition(parser):
    start = parser.token.start
    expect_keyword(parser, 'union')
    return ast.UnionTypeDefinition(name=parse_name(parser), directives=parse_directives(parser), types=expect(parser, TokenKind.EQUALS) and parse_union_members(parser), loc=loc(parser, start))