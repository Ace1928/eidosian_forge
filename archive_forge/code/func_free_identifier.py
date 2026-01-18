import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def free_identifier(self, lineno: t.Optional[int]=None) -> nodes.InternalName:
    """Return a new free identifier as :class:`~jinja2.nodes.InternalName`."""
    self._last_identifier += 1
    rv = object.__new__(nodes.InternalName)
    nodes.Node.__init__(rv, f'fi{self._last_identifier}', lineno=lineno)
    return rv