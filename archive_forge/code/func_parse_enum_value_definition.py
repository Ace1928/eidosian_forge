from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_enum_value_definition(parser):
    start = parser.token.start
    return ast.EnumValueDefinition(name=parse_name(parser), directives=parse_directives(parser), loc=loc(parser, start))