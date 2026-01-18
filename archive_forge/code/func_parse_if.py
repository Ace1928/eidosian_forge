import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_if(self) -> nodes.If:
    """Parse an if construct."""
    node = result = nodes.If(lineno=self.stream.expect('name:if').lineno)
    while True:
        node.test = self.parse_tuple(with_condexpr=False)
        node.body = self.parse_statements(('name:elif', 'name:else', 'name:endif'))
        node.elif_ = []
        node.else_ = []
        token = next(self.stream)
        if token.test('name:elif'):
            node = nodes.If(lineno=self.stream.current.lineno)
            result.elif_.append(node)
            continue
        elif token.test('name:else'):
            result.else_ = self.parse_statements(('name:endif',), drop_needle=True)
        break
    return result