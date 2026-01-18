import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_autoescape(self) -> nodes.Scope:
    node = nodes.ScopedEvalContextModifier(lineno=next(self.stream).lineno)
    node.options = [nodes.Keyword('autoescape', self.parse_expression())]
    node.body = self.parse_statements(('name:endautoescape',), drop_needle=True)
    return nodes.Scope([node])