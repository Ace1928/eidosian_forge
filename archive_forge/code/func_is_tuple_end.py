import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def is_tuple_end(self, extra_end_rules: t.Optional[t.Tuple[str, ...]]=None) -> bool:
    """Are we at the end of a tuple?"""
    if self.stream.current.type in ('variable_end', 'block_end', 'rparen'):
        return True
    elif extra_end_rules is not None:
        return self.stream.current.test_any(extra_end_rules)
    return False