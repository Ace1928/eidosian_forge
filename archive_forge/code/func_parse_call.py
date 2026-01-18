import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_call(self, node: nodes.Expr) -> nodes.Call:
    token = self.stream.current
    args, kwargs, dyn_args, dyn_kwargs = self.parse_call_args()
    return nodes.Call(node, args, kwargs, dyn_args, dyn_kwargs, lineno=token.lineno)