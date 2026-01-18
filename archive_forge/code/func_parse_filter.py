import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_filter(self, node: t.Optional[nodes.Expr], start_inline: bool=False) -> t.Optional[nodes.Expr]:
    while self.stream.current.type == 'pipe' or start_inline:
        if not start_inline:
            next(self.stream)
        token = self.stream.expect('name')
        name = token.value
        while self.stream.current.type == 'dot':
            next(self.stream)
            name += '.' + self.stream.expect('name').value
        if self.stream.current.type == 'lparen':
            args, kwargs, dyn_args, dyn_kwargs = self.parse_call_args()
        else:
            args = []
            kwargs = []
            dyn_args = dyn_kwargs = None
        node = nodes.Filter(node, name, args, kwargs, dyn_args, dyn_kwargs, lineno=token.lineno)
        start_inline = False
    return node