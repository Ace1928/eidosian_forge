from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_schema_definition(parser):
    start = parser.token.start
    expect_keyword(parser, 'schema')
    directives = parse_directives(parser)
    operation_types = many(parser, TokenKind.BRACE_L, parse_operation_type_definition, TokenKind.BRACE_R)
    return ast.SchemaDefinition(directives=directives, operation_types=operation_types, loc=loc(parser, start))