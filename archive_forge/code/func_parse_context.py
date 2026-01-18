import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_context() -> bool:
    if self.stream.current.value in {'with', 'without'} and self.stream.look().test('name:context'):
        node.with_context = next(self.stream).value == 'with'
        self.stream.skip()
        return True
    return False