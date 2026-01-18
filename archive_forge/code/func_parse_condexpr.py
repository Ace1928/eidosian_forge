import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_condexpr(self) -> nodes.Expr:
    lineno = self.stream.current.lineno
    expr1 = self.parse_or()
    expr3: t.Optional[nodes.Expr]
    while self.stream.skip_if('name:if'):
        expr2 = self.parse_or()
        if self.stream.skip_if('name:else'):
            expr3 = self.parse_condexpr()
        else:
            expr3 = None
        expr1 = nodes.CondExpr(expr2, expr1, expr3, lineno=lineno)
        lineno = self.stream.current.lineno
    return expr1