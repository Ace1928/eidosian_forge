import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_primary(self) -> nodes.Expr:
    token = self.stream.current
    node: nodes.Expr
    if token.type == 'name':
        if token.value in ('true', 'false', 'True', 'False'):
            node = nodes.Const(token.value in ('true', 'True'), lineno=token.lineno)
        elif token.value in ('none', 'None'):
            node = nodes.Const(None, lineno=token.lineno)
        else:
            node = nodes.Name(token.value, 'load', lineno=token.lineno)
        next(self.stream)
    elif token.type == 'string':
        next(self.stream)
        buf = [token.value]
        lineno = token.lineno
        while self.stream.current.type == 'string':
            buf.append(self.stream.current.value)
            next(self.stream)
        node = nodes.Const(''.join(buf), lineno=lineno)
    elif token.type in ('integer', 'float'):
        next(self.stream)
        node = nodes.Const(token.value, lineno=token.lineno)
    elif token.type == 'lparen':
        next(self.stream)
        node = self.parse_tuple(explicit_parentheses=True)
        self.stream.expect('rparen')
    elif token.type == 'lbracket':
        node = self.parse_list()
    elif token.type == 'lbrace':
        node = self.parse_dict()
    else:
        self.fail(f'unexpected {describe_token(token)!r}', token.lineno)
    return node