from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_document(parser):
    start = parser.token.start
    definitions = []
    while True:
        definitions.append(parse_definition(parser))
        if skip(parser, TokenKind.EOF):
            break
    return ast.Document(definitions=definitions, loc=loc(parser, start))