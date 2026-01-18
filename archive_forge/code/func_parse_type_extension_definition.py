from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_type_extension_definition(parser):
    start = parser.token.start
    expect_keyword(parser, 'extend')
    return ast.TypeExtensionDefinition(definition=parse_object_type_definition(parser), loc=loc(parser, start))