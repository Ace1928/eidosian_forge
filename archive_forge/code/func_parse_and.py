import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_and(self) -> nodes.Expr:
    lineno = self.stream.current.lineno
    left = self.parse_not()
    while self.stream.skip_if('name:and'):
        right = self.parse_not()
        left = nodes.And(left, right, lineno=lineno)
        lineno = self.stream.current.lineno
    return left