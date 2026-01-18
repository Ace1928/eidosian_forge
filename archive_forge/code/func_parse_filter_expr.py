import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_filter_expr(self, node: nodes.Expr) -> nodes.Expr:
    while True:
        token_type = self.stream.current.type
        if token_type == 'pipe':
            node = self.parse_filter(node)
        elif token_type == 'name' and self.stream.current.value == 'is':
            node = self.parse_test(node)
        elif token_type == 'lparen':
            node = self.parse_call(node)
        else:
            break
    return node