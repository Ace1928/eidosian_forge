import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def flush_data() -> None:
    if data_buffer:
        lineno = data_buffer[0].lineno
        body.append(nodes.Output(data_buffer[:], lineno=lineno))
        del data_buffer[:]