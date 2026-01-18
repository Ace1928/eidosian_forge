from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_field_definition(parser):
    start = parser.token.start
    return ast.FieldDefinition(name=parse_name(parser), arguments=parse_argument_defs(parser), type=expect(parser, TokenKind.COLON) and parse_type(parser), directives=parse_directives(parser), loc=loc(parser, start))