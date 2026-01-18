import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_include(self) -> nodes.Include:
    node = nodes.Include(lineno=next(self.stream).lineno)
    node.template = self.parse_expression()
    if self.stream.current.test('name:ignore') and self.stream.look().test('name:missing'):
        node.ignore_missing = True
        self.stream.skip(2)
    else:
        node.ignore_missing = False
    return self.parse_import_context(node, True)