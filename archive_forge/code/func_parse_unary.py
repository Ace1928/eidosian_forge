import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_unary(self, with_filter: bool=True) -> nodes.Expr:
    token_type = self.stream.current.type
    lineno = self.stream.current.lineno
    node: nodes.Expr
    if token_type == 'sub':
        next(self.stream)
        node = nodes.Neg(self.parse_unary(False), lineno=lineno)
    elif token_type == 'add':
        next(self.stream)
        node = nodes.Pos(self.parse_unary(False), lineno=lineno)
    else:
        node = self.parse_primary()
    node = self.parse_postfix(node)
    if with_filter:
        node = self.parse_filter_expr(node)
    return node