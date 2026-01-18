from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def many(parser, open_kind, parse_fn, close_kind):
    """Returns a non-empty list of parse nodes, determined by
    the parse_fn. This list begins with a lex token of openKind
    and ends with a lex token of closeKind. Advances the parser
    to the next lex token after the closing token."""
    expect(parser, open_kind)
    nodes = [parse_fn(parser)]
    while not skip(parser, close_kind):
        nodes.append(parse_fn(parser))
    return nodes