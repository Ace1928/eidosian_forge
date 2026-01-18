import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_extends(self) -> nodes.Extends:
    node = nodes.Extends(lineno=next(self.stream).lineno)
    node.template = self.parse_expression()
    return node