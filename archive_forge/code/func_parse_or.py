import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_or(self) -> nodes.Expr:
    lineno = self.stream.current.lineno
    left = self.parse_and()
    while self.stream.skip_if('name:or'):
        right = self.parse_and()
        left = nodes.Or(left, right, lineno=lineno)
        lineno = self.stream.current.lineno
    return left