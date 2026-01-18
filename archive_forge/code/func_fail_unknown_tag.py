import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def fail_unknown_tag(self, name: str, lineno: t.Optional[int]=None) -> 'te.NoReturn':
    """Called if the parser encounters an unknown tag.  Tries to fail
        with a human readable error message that could help to identify
        the problem.
        """
    self._fail_ut_eof(name, self._end_token_stack, lineno)