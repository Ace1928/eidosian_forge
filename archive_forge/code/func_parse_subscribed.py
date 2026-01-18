import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_subscribed(self) -> nodes.Expr:
    lineno = self.stream.current.lineno
    args: t.List[t.Optional[nodes.Expr]]
    if self.stream.current.type == 'colon':
        next(self.stream)
        args = [None]
    else:
        node = self.parse_expression()
        if self.stream.current.type != 'colon':
            return node
        next(self.stream)
        args = [node]
    if self.stream.current.type == 'colon':
        args.append(None)
    elif self.stream.current.type not in ('rbracket', 'comma'):
        args.append(self.parse_expression())
    else:
        args.append(None)
    if self.stream.current.type == 'colon':
        next(self.stream)
        if self.stream.current.type not in ('rbracket', 'comma'):
            args.append(self.parse_expression())
        else:
            args.append(None)
    else:
        args.append(None)
    return nodes.Slice(*args, lineno=lineno)