from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_value_literal(parser, is_const):
    token = parser.token
    if token.kind == TokenKind.BRACKET_L:
        return parse_list(parser, is_const)
    elif token.kind == TokenKind.BRACE_L:
        return parse_object(parser, is_const)
    elif token.kind == TokenKind.INT:
        advance(parser)
        return ast.IntValue(value=token.value, loc=loc(parser, token.start))
    elif token.kind == TokenKind.FLOAT:
        advance(parser)
        return ast.FloatValue(value=token.value, loc=loc(parser, token.start))
    elif token.kind == TokenKind.STRING:
        advance(parser)
        return ast.StringValue(value=token.value, loc=loc(parser, token.start))
    elif token.kind == TokenKind.NAME:
        if token.value in ('true', 'false'):
            advance(parser)
            return ast.BooleanValue(value=token.value == 'true', loc=loc(parser, token.start))
        if token.value != 'null':
            advance(parser)
            return ast.EnumValue(value=token.value, loc=loc(parser, token.start))
    elif token.kind == TokenKind.DOLLAR:
        if not is_const:
            return parse_variable(parser)
    raise unexpected(parser)