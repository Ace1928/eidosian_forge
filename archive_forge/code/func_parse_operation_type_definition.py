from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_operation_type_definition(parser):
    start = parser.token.start
    operation = parse_operation_type(parser)
    expect(parser, TokenKind.COLON)
    return ast.OperationTypeDefinition(operation=operation, type=parse_named_type(parser), loc=loc(parser, start))