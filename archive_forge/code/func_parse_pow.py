import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_pow(self) -> nodes.Expr:
    lineno = self.stream.current.lineno
    left = self.parse_unary()
    while self.stream.current.type == 'pow':
        next(self.stream)
        right = self.parse_unary()
        left = nodes.Pow(left, right, lineno=lineno)
        lineno = self.stream.current.lineno
    return left