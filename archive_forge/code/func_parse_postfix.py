import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_postfix(self, node: nodes.Expr) -> nodes.Expr:
    while True:
        token_type = self.stream.current.type
        if token_type == 'dot' or token_type == 'lbracket':
            node = self.parse_subscript(node)
        elif token_type == 'lparen':
            node = self.parse_call(node)
        else:
            break
    return node