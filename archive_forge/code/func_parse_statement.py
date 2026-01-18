import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_statement(self) -> t.Union[nodes.Node, t.List[nodes.Node]]:
    """Parse a single statement."""
    token = self.stream.current
    if token.type != 'name':
        self.fail('tag name expected', token.lineno)
    self._tag_stack.append(token.value)
    pop_tag = True
    try:
        if token.value in _statement_keywords:
            f = getattr(self, f'parse_{self.stream.current.value}')
            return f()
        if token.value == 'call':
            return self.parse_call_block()
        if token.value == 'filter':
            return self.parse_filter_block()
        ext = self.extensions.get(token.value)
        if ext is not None:
            return ext(self)
        self._tag_stack.pop()
        pop_tag = False
        self.fail_unknown_tag(token.value, token.lineno)
    finally:
        if pop_tag:
            self._tag_stack.pop()